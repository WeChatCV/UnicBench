"""
UnicBench Score Calculator
分数统计工具代码

使用方法:
python calculate_scores.py \
    --save_dir /path/to/results \
    --edit_model_name qwen-image-edit \
    --vlm_model_name gpt-4.1 \
    --languages en cn
"""

import os
import json
import argparse
import logging
from typing import Dict, List, Any
import numpy as np


def setup_logging(log_level: str = "INFO") -> None:
    """配置日志"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('calculate_scores.log')
        ]
    )


def compute_stats_for_group(results: List[Dict], group_name: str) -> Dict[str, float]:
    """计算一组结果的统计信息"""
    if not results:
        return {}
    
    stats = {
        "count": len(results),
        "if_mean": np.mean([r["if_score"] for r in results]),
        "nc_mean": np.mean([r["nc_score"] for r in results]),
        "vq_mean": np.mean([r["vq_score"] for r in results]),
        "overall_mean": np.mean([r["overall_score"] for r in results])
    }
    
    # 添加RA统计
    ra_scores = [r["ra_score"] for r in results if r["ra_score"] is not None]
    if ra_scores:
        stats["ra_mean"] = np.mean(ra_scores)
        stats["ra_count"] = len(ra_scores)
    
    return stats


def print_statistics_summary(stats: Dict, language: str) -> None:
    """打印统计摘要"""
    logging.info(f"\n{'='*60}")
    logging.info(f"Statistics Summary - {language.upper()}")
    logging.info(f"{'='*60}")
    
    # 整体统计
    overall = stats["overall"]
    logging.info(f"\nOverall (n={overall['count']}):")
    logging.info(f"  IF:      {overall['if_mean']:.4f}")
    logging.info(f"  NC:      {overall['nc_mean']:.4f}")
    logging.info(f"  VQ:      {overall['vq_mean']:.4f}")
    if "ra_mean" in overall:
        logging.info(f"  RA:      {overall['ra_mean']:.4f} (n={overall['ra_count']})")
    logging.info(f"  Overall: {overall['overall_mean']:.4f}")
    
    # Task级别统计
    logging.info(f"\nBy Task:")
    for task, task_stats in stats["by_task"].items():
        logging.info(f"  {task} (n={task_stats['count']}):")
        logging.info(f"    IF: {task_stats['if_mean']:.4f}, NC: {task_stats['nc_mean']:.4f}, "
                    f"VQ: {task_stats['vq_mean']:.4f}, Overall: {task_stats['overall_mean']:.4f}")
        if "ra_mean" in task_stats:
            logging.info(f"    RA: {task_stats['ra_mean']:.4f}")
    
    # Subtask级别统计
    logging.info(f"\nBy Subtask:")
    for subtask, subtask_stats in stats["by_subtask"].items():
        logging.info(f"  {subtask} (n={subtask_stats['count']}):")
        logging.info(f"    IF: {subtask_stats['if_mean']:.4f}, NC: {subtask_stats['nc_mean']:.4f}, "
                    f"VQ: {subtask_stats['vq_mean']:.4f}, Overall: {subtask_stats['overall_mean']:.4f}")
        if "ra_mean" in subtask_stats:
            logging.info(f"    RA: {subtask_stats['ra_mean']:.4f}")


def compute_statistics(
    save_dir: str,
    edit_model_name: str,
    vlm_model_name: str,
    languages: List[str]
) -> Dict[str, Any]:
    """计算所有评估结果的统计信息"""
    logging.info("="*60)
    logging.info("Computing Statistics")
    logging.info("="*60)
    
    all_statistics = {}
    
    for language in languages:
        logging.info(f"\nComputing statistics for language: {language}")
        
        # 收集所有结果
        all_results = []
        eval_base_dir = os.path.join(
            save_dir,
            edit_model_name,
            "eval_output",
            vlm_model_name
        )
        
        # 查找所有结果文件
        if not os.path.exists(eval_base_dir):
            logging.warning(f"Evaluation directory not found: {eval_base_dir}")
            continue
            
        for subtask_file in os.listdir(eval_base_dir):
            if subtask_file.endswith(f"{language}_results.jsonl"):
                subtask_path = os.path.join(eval_base_dir, subtask_file)
                if os.path.exists(subtask_path):
                    with open(subtask_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            all_results.append(json.loads(line.strip()))
        
        if not all_results:
            logging.warning(f"No results found for language: {language}")
            continue
        
        # 计算整体统计
        overall_stats = compute_stats_for_group(all_results, "Overall")
        
        # 计算Task级别统计
        task_stats = {}
        tasks = set([r["task"] for r in all_results])
        for task in tasks:
            task_results = [r for r in all_results if r["task"] == task]
            task_stats[task] = compute_stats_for_group(task_results, task)
        
        # 计算Subtask级别统计
        subtask_stats = {}
        subtasks = set([r["subtask"] for r in all_results])
        for subtask in subtasks:
            subtask_results = [r for r in all_results if r["subtask"] == subtask]
            subtask_stats[subtask] = compute_stats_for_group(subtask_results, subtask)
        
        # 存储统计信息
        all_statistics[language] = {
            "overall": overall_stats,
            "by_task": task_stats,
            "by_subtask": subtask_stats
        }
        
        # 保存统计信息到文件
        stats_dir = os.path.join(
            save_dir,
            edit_model_name,
            "eval_output",
            vlm_model_name,
            "statistics"
        )
        os.makedirs(stats_dir, exist_ok=True)
        
        stats_file = os.path.join(stats_dir, f"{language}_statistics.json")
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(all_statistics[language], f, indent=2, ensure_ascii=False)
        
        logging.info(f"Saved statistics to {stats_file}")
        
        # 打印摘要
        print_statistics_summary(all_statistics[language], language)
    
    return all_statistics


def parse_arguments() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="UnicBench Score Calculator")
    
    parser.add_argument('--save_dir', type=str, required=True,
                       help='结果保存根目录')
    parser.add_argument('--edit_model_name', type=str, required=True,
                       help='编辑模型名称')
    parser.add_argument('--vlm_model_name', type=str, required=True,
                       help='VLM评估模型名称')
    parser.add_argument('--languages', type=str, nargs='+', default=['en'],
                       choices=['en', 'cn'],
                       help='测试语言 (en, cn, 或两者)')
    parser.add_argument('--log_level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='日志级别')
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_arguments()
    
    # 配置日志
    setup_logging(args.log_level)
    
    logging.info("="*60)
    logging.info("Starting UnicBench Score Calculator")
    logging.info("="*60)
    
    # 计算统计信息
    compute_statistics(
        save_dir=args.save_dir,
        edit_model_name=args.edit_model_name,
        vlm_model_name=args.vlm_model_name,
        languages=args.languages
    )
    
    logging.info("\n" + "="*60)
    logging.info("Score calculation completed!")
    logging.info("="*60)


if __name__ == "__main__":
    main()
