"""
UnicBench Evaluation Pipeline
专用于评测和统计分数的代码（不包含编辑图像）

使用方法:
python eval_pipeline.py \
    --data_path ../data/test_data.jsonl \
    --save_dir /path/to/results \
    --edit_model_name qwen-image-edit \
    --vlm_model_name gpt-4.1 \
    --languages en cn \
    --num_workers 4
"""

import os
import json
import argparse
import random
import sys
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data import prompts


# ===================== utils =====================

def set_seed(seed: int) -> None:
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logging(log_level: str = "INFO") -> None:
    """配置日志"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('eval_benchmark.log')
        ]
    )


# ===================== VLM Init =====================

class VLMModel:
    """VLM评估模型类"""
    
    def __init__(self, model_name: str = "gpt-4.1", model_path: Optional[str] = None):
        self.model_name = model_name
        self.model_path = model_path

        if model_name == "gpt-4.1":
            from models.ichat_gpt4o import Ichat_GPT4o
            self.model = Ichat_GPT4o(model="gpt-4.1")
        elif model_name == "qwen25vl":
            from models.qwen_vl_base import QwenVL
            self.model = QwenVL(model_path=model_path, tensor_parallel_size=8)
        else:
            raise ValueError(f"Unsupported VLM model: {model_name}")
        
        logging.info(f"Initialized VLMModel: {model_name}")
    
    def prepare_prompt(self, text_prompt: str, images: List[Image.Image]) -> Any:
        """准备prompt"""
        return self.model.prepare_prompt(images, text_prompt)
    
    def forward(self, final_prompt: Any) -> str:
        """VLM前向推理"""
        return self.model.forward(final_prompt)


# ===================== UnicBenchEvaluator =====================

class UnicBenchEvaluator:
    """UnicBench评测器"""
    
    def __init__(
        self,
        data_path: str,
        save_dir: str,
        edit_model_name: str,
        vlm_model: VLMModel,
        languages: List[str] = ["en"],
        num_workers: int = 1
    ):
        """
        初始化评测器
        
        Args:
            data_path: 测试数据jsonl文件路径
            save_dir: 结果保存根目录
            edit_model_name: 编辑模型名称
            vlm_model: VLM评估模型实例
            languages: 语言列表 ["en", "cn"]
            num_workers: 并行评估的worker数量
        """
        self.data_path = data_path
        self.save_dir = save_dir
        self.edit_model_name = edit_model_name
        self.vlm_model = vlm_model
        self.languages = languages
        self.num_workers = num_workers
        
        # 加载评估prompts
        self.if_prompt = prompts._prompts_if
        self.nc_prompt = prompts._prompts_nc
        self.vq_prompt = prompts._prompts_vq
        self.ra_prompt = prompts._prompts_ra

        # 加载测试数据
        self.data_list = self._load_data()
        
        logging.info(f"Loaded {len(self.data_list)} test samples")
        logging.info(f"Languages: {languages}")
        logging.info(f"Number of workers: {num_workers}")
    
    def _load_data(self) -> List[Dict]:
        """加载测试数据"""
        data_list = []
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line in f:
                data_list.append(json.loads(line.strip()))
        return data_list
    
    def _organize_data_by_task_subtask(self) -> Dict[str, Dict[str, List[Dict]]]:
        """按task和subtask组织数据"""
        organized = {}
        for sample in self.data_list:
            task = sample["task"]
            subtask = sample["subtask"]
            
            if task not in organized:
                organized[task] = {}
            if subtask not in organized[task]:
                organized[task][subtask] = []
            
            organized[task][subtask].append(sample)
        
        return organized
    
    def _parse_vlm_response(self, response: str) -> Dict[str, Any]:
        """解析VLM响应"""
        response = response.replace('```json', '').replace('```', '').strip()
        response = response.replace('(', '[').replace(')', ']')
        
        try:
            result = json.loads(response)
            return {
                "score": float(result.get("score", 0)),
                "reason": result.get("reason", "")
            }
        except:
            logging.warning(f"Failed to parse VLM response: {response[:100]}")
            return {"score": 0.0, "reason": "Parse error"}
    
    def _evaluate_single_sample(
        self,
        sample: Dict,
        language: str,
        original_image: Image.Image,
        edited_image: Image.Image,
        edited_image_path: str
    ) -> Dict[str, Any]:
        """评估单个样本"""
        instruction = sample[language]
        images = [original_image, edited_image]
        
        result = {
            "key": sample["key"],
            "task": sample["task"],
            "subtask": sample["subtask"],
            "instruction": instruction,
            "instruction_language": language,
            "edited_image_path": edited_image_path
        }
        
        # 评估IF (Instruction Following)
        if_prompt_text = self.if_prompt.replace("<instruction>", instruction)
        if_final_prompt = self.vlm_model.prepare_prompt(if_prompt_text, images)
        if_response = self.vlm_model.forward(if_final_prompt)
        if_result = self._parse_vlm_response(if_response)
        result["if_score"] = if_result["score"]
        result["if_reason"] = if_result["reason"]
        
        # 评估NC (Non-edited region Consistency)
        nc_prompt_text = self.nc_prompt.replace("<instruction>", instruction)
        nc_final_prompt = self.vlm_model.prepare_prompt(nc_prompt_text, images)
        nc_response = self.vlm_model.forward(nc_final_prompt)
        nc_result = self._parse_vlm_response(nc_response)
        result["nc_score"] = nc_result["score"]
        result["nc_reason"] = nc_result["reason"]
        
        # 评估VQ (Visual Quality)
        vq_prompt_text = self.vq_prompt.replace("<instruction>", instruction)
        vq_final_prompt = self.vlm_model.prepare_prompt(vq_prompt_text, images)
        vq_response = self.vlm_model.forward(vq_final_prompt)
        vq_result = self._parse_vlm_response(vq_response)
        result["vq_score"] = vq_result["score"]
        result["vq_reason"] = vq_result["reason"]
        
        # 评估RA (Reasoning Accuracy) - 仅用于Reasoning Editing任务
        if sample["task"] == "Reasoning Editing" and "reasoning_points" in sample:
            reasoning_points = json.dumps(sample["reasoning_points"])
            ra_prompt_text = self.ra_prompt.replace("<instruction>", instruction).replace(
                "<reasoning_points>", reasoning_points
            )
            ra_final_prompt = self.vlm_model.prepare_prompt(ra_prompt_text, images)
            ra_response = self.vlm_model.forward(ra_final_prompt)
            ra_result = self._parse_vlm_response(ra_response)
            result["ra_score"] = ra_result["score"]
            result["ra_reason"] = ra_result["reason"]
            
            # 计算总体得分 (4个指标)
            scores = [if_result["score"], nc_result["score"], vq_result["score"], ra_result["score"]]
            result["overall_score"] = np.power(np.prod(scores), 1.0/4.0)
        else:
            result["ra_score"] = None
            result["ra_reason"] = None
            
            # 计算总体得分 (3个指标)
            scores = [if_result["score"], nc_result["score"], vq_result["score"]]
            result["overall_score"] = np.power(np.prod(scores), 1.0/3.0)
        
        return result
    
    def _evaluate_single_wrapper(self, args: tuple) -> Optional[Dict[str, Any]]:
        """评估单个样本的包装函数"""
        sample, language, original_image, edited_image, edited_image_path = args
        try:
            return self._evaluate_single_sample(sample, language, original_image, edited_image, edited_image_path)
        except Exception as e:
            logging.error(f"Error evaluating sample {sample['key']}: {e}")
            return None

    def evaluate(self) -> None:
        """执行评估"""
        logging.info("="*60)
        logging.info("Starting Evaluation Phase")
        logging.info("="*60)
        
        organized_data = self._organize_data_by_task_subtask()
        vlm_name = self.vlm_model.model_name
        
        for language in self.languages:
            logging.info(f"{'='*60}")
            logging.info(f"Evaluating Language: {language.upper()}")
            logging.info(f"{'='*60}")
            
            for task_idx, (task, subtasks) in enumerate(organized_data.items(), 1):
                logging.info(f"\n[Task {task_idx}/{len(organized_data)}] {task}")
                
                for subtask_idx, (subtask, samples) in enumerate(subtasks.items(), 1):
                    logging.info(f"  [Subtask {subtask_idx}/{len(subtasks)}] {subtask} - {len(samples)} samples")
                    
                    # 准备路径
                    subtask_name = '_'.join(subtask.split())
                    img_dir = os.path.join(self.save_dir, self.edit_model_name, subtask_name, language)
                    eval_dir = os.path.join(self.save_dir, self.edit_model_name, "eval_output", vlm_name)
                    os.makedirs(eval_dir, exist_ok=True)
                    eval_file = os.path.join(eval_dir, f"{subtask_name}_{language}_results.jsonl")

                    # 检查是否已评估
                    if os.path.exists(eval_file):
                        logging.info(f"  Skipping {subtask} as it has already been evaluated")
                        continue
                    
                    # 准备评估任务
                    eval_tasks = []
                    for sample in samples:
                        try:
                            original_image = Image.open(sample["image_path"]).convert("RGB")
                            key = sample["key"]
                            img_path = os.path.join(img_dir, f"{key}.png")
                            
                            if not os.path.exists(img_path):
                                logging.warning(f"Edited image not found: {img_path}")
                                continue
                            
                            edited_image = Image.open(img_path).convert("RGB")
                            eval_tasks.append((sample, language, original_image, edited_image, img_path))
                            
                        except Exception as e:
                            logging.error(f"Error loading images for sample {sample['key']}: {e}")
                            continue
                    
                    if not eval_tasks:
                        logging.warning(f"  No valid samples to evaluate for {subtask}")
                        continue
                    
                    # 执行评估
                    results = []
                    
                    if "gpt" in vlm_name.lower() and self.num_workers > 1:
                        # GPT模型使用多线程
                        logging.info(f"  Using multi-threading with {self.num_workers} workers")
                        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                            futures = [executor.submit(self._evaluate_single_wrapper, task) for task in eval_tasks]
                            
                            for future in tqdm(as_completed(futures), total=len(futures), 
                                             desc=f"  Evaluating {task}/{subtask}", leave=False):
                                result = future.result()
                                if result is not None:
                                    results.append(result)
                    else:
                        # 单线程评估
                        logging.info("  Using single-threaded evaluation")
                        for task_data in tqdm(eval_tasks, desc=f"  Evaluating {task}/{subtask}", leave=False):
                            result = self._evaluate_single_wrapper(task_data)
                            if result is not None:
                                results.append(result)
                    
                    # 保存结果
                    if results:
                        with open(eval_file, 'w', encoding='utf-8') as f:
                            for item in results:
                                f.write(json.dumps(item, ensure_ascii=False) + '\n')
                        
                        logging.info(f"  Saved {len(results)} results to {eval_file}")
        
        logging.info("\n" + "="*60)
        logging.info("Evaluation completed!")
        logging.info("="*60)


# ===================== UnicBenchStatistics =====================

class UnicBenchStatistics:
    """UnicBench统计器"""
    
    def __init__(
        self,
        save_dir: str,
        edit_model_name: str,
        vlm_model_name: str,
        languages: List[str] = ["en"]
    ):
        """
        初始化统计器
        
        Args:
            save_dir: 结果保存根目录
            edit_model_name: 编辑模型名称
            vlm_model_name: VLM模型名称
            languages: 语言列表
        """
        self.save_dir = save_dir
        self.edit_model_name = edit_model_name
        self.vlm_model_name = vlm_model_name
        self.languages = languages
    
    def _compute_stats_for_group(self, results: List[Dict], group_name: str) -> Dict[str, float]:
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
    
    def _print_statistics_summary(self, stats: Dict, language: str) -> None:
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
    
    def compute_statistics(self) -> Dict[str, Any]:
        """计算所有评估结果的统计信息"""
        logging.info("="*60)
        logging.info("Computing Statistics")
        logging.info("="*60)
        
        all_statistics = {}
        
        for language in self.languages:
            logging.info(f"\nComputing statistics for language: {language}")
            
            # 收集所有结果
            all_results = []
            eval_base_dir = os.path.join(
                self.save_dir,
                self.edit_model_name,
                "eval_output",
                self.vlm_model_name
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
            overall_stats = self._compute_stats_for_group(all_results, "Overall")
            
            # 计算Task级别统计
            task_stats = {}
            tasks = set([r["task"] for r in all_results])
            for task in tasks:
                task_results = [r for r in all_results if r["task"] == task]
                task_stats[task] = self._compute_stats_for_group(task_results, task)
            
            # 计算Subtask级别统计
            subtask_stats = {}
            subtasks = set([r["subtask"] for r in all_results])
            for subtask in subtasks:
                subtask_results = [r for r in all_results if r["subtask"] == subtask]
                subtask_stats[subtask] = self._compute_stats_for_group(subtask_results, subtask)
            
            # 存储统计信息
            all_statistics[language] = {
                "overall": overall_stats,
                "by_task": task_stats,
                "by_subtask": subtask_stats
            }
            
            # 保存统计信息到文件
            stats_dir = os.path.join(
                self.save_dir,
                self.edit_model_name,
                "eval_output",
                self.vlm_model_name,
                "statistics"
            )
            os.makedirs(stats_dir, exist_ok=True)
            
            stats_file = os.path.join(stats_dir, f"{language}_statistics.json")
            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump(all_statistics[language], f, indent=2, ensure_ascii=False)
            
            logging.info(f"Saved statistics to {stats_file}")
            
            # 打印摘要
            self._print_statistics_summary(all_statistics[language], language)
        
        return all_statistics


# ===================== MAIN =====================

def parse_arguments() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="UnicBench Evaluation Pipeline")
    
    # 数据参数
    parser.add_argument('--data_path', type=str, required=True,
                       help='测试数据jsonl文件路径')
    parser.add_argument('--save_dir', type=str, required=True,
                       help='结果保存根目录')
    
    # 模型参数
    parser.add_argument('--edit_model_name', type=str, required=True,
                       help='编辑模型名称')
    parser.add_argument('--vlm_model_name', type=str, default='gpt-4.1',
                       help='VLM评估模型名称')
    parser.add_argument('--vlm_model_path', type=str, default=None,
                       help='VLM模型权重路径 (本地模型)')
    
    # 语言参数
    parser.add_argument('--languages', type=str, nargs='+', default=['en'],
                       choices=['en', 'cn'],
                       help='测试语言 (en, cn, 或两者)')
    
    # 执行参数
    parser.add_argument('--skip_evaluation', action='store_true',
                       help='跳过评估阶段，仅执行统计')
    parser.add_argument('--num_workers', type=int, default=1,
                       help='并行评估的worker数量 (仅用于GPT模型)')
    
    # 日志参数
    parser.add_argument('--log_level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='日志级别')
    
    # 随机种子
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_arguments()
    
    # 配置日志
    setup_logging(args.log_level)
    
    # 设置随机种子
    set_seed(args.seed)
    
    logging.info("="*60)
    logging.info("Starting UnicBench Evaluation Pipeline")
    logging.info("="*60)
    
    start_time = time.time()
    
    # 步骤1: 评估
    if not args.skip_evaluation:
        # 初始化VLM模型
        vlm_model = VLMModel(model_name=args.vlm_model_name, model_path=args.vlm_model_path)
        
        # 初始化评估器
        evaluator = UnicBenchEvaluator(
            data_path=args.data_path,
            save_dir=args.save_dir,
            edit_model_name=args.edit_model_name,
            vlm_model=vlm_model,
            languages=args.languages,
            num_workers=args.num_workers
        )
        
        # 执行评估
        evaluator.evaluate()
    
    # 步骤2: 统计
    statistics = UnicBenchStatistics(
        save_dir=args.save_dir,
        edit_model_name=args.edit_model_name,
        vlm_model_name=args.vlm_model_name,
        languages=args.languages
    )
    
    # 计算统计信息
    statistics.compute_statistics()
    
    elapsed_time = time.time() - start_time
    logging.info(f"\n{'='*60}")
    logging.info(f"Pipeline completed in {elapsed_time:.2f} seconds")
    logging.info(f"{'='*60}")


if __name__ == "__main__":
    main()
