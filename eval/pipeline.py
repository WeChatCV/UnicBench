import os
import json
import argparse
import random
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import logging
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time

import prompts

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class EditModel:
    """Base class for image editing models"""
    def __init__(self, model_name: str = "qwen-image-edit", model_path: Optional[str] = None):
        """
        Initialize editing model
        
        Args:
            model_name: Name of the editing model
        """
        self.model_name = model_name
        if model_name == "qwen-image-edit":
            from models.qwen_editting import QwenImageEdit
            self.model = QwenImageEdit(model_path="/mnt/shenzhen2cephfs/mm-base-vision/kotisye/pretrain/Qwen-Image-Edit")
        elif model_name == "qwen-image-edit-2509":
            from models.qwen_editting import QwenImageEdit2509
            self.model = QwenImageEdit2509(model_path="/mnt/shenzhen2cephfs/mm-base-vision/kotisye/pretrain/Qwen-Image-Edit-2509")
        elif model_name == "flux.1-kontext":
            from models.flux_editting import FluxKontextEditor
            self.model = FluxKontextEditor(model_path="/mnt/shenzhen2cephfs/mm-base-vision/kotisye/pretrain/FLUX.1-Kontext-dev")
        elif model_name == "omnigen2":
            from models.OmniGen2.modeling_omnigen2 import OmniGen2
            self.model = OmniGen2(model_path="/mnt/shenzhen2cephfs/mm-base-vision/kotisye/pretrain/OmniGen2")
        elif model_name == "step1x-edit-v1p1":
            from models.Step1XEdit.modeling_step1xedit import Step1XEditModel
            self.model = Step1XEditModel(model_path="/mnt/shenzhen2cephfs/mm-base-vision/kotisye/pretrain/Step1X-Edit-v1p1-diffusers")
        elif model_name == "bagel":
            from models.Bagel.modeling_bagel import BagelModel
            self.model = BagelModel(model_path="/mnt/shenzhen2cephfs/mm-base-vision/kotisye/pretrain/BAGEL-7B-MoT")
        elif model_name == "instruct-pix2pix":
            from models.instructp2p_editing import InstructPix2PixModel
            self.model = InstructPix2PixModel(model_path="/mnt/shenzhen2cephfs/mm-base-vision/kotisye/pretrain/instruct-pix2pix")
        elif model_name == "magicbrush":
            from models.magicbrush_editing import MagicBrush
            self.model = MagicBrush(model_path="/mnt/shenzhen2cephfs/mm-base-vision/kotisye/pretrain/magicbrush-jul7")
        elif model_name == "uniworldv1":
            from models.UniWorldv1.modeling_uniworldv1 import UniWorldv1Model
            self.model = UniWorldv1Model(pretrained_lvlm_name_or_path="/mnt/shenzhen2cephfs/mm-base-vision/kotisye/pretrain/UniWorld-V1")
        elif model_name == "nano_banana":
            from models.nano_banana_ichat import GeminiImageGenerator
            API_BASE_URL = os.environ.get("ICHAT_API_BASE_URL", "http://ichat.woa.com/api")
            APP_ID = os.environ.get("ICHAT_APP_ID", "appyvzx1xp5h2b7wnqh")
            APP_SECRET = os.environ.get("ICHAT_APP_SECRET", "PwkBNPZMbVEpJcYVtcYZAXkYvsldGrpM")
            SOURCE = os.environ.get("ICHAT_SOURCE", "nano-banana-demo")
            
            self.model = GeminiImageGenerator(
                api_base_url=API_BASE_URL,
                app_id=APP_ID,
                app_secret=APP_SECRET,
                source=SOURCE
            )
        elif model_name == "gpt-image-1":
            from models.gptimage1_ichat import GPTImageGenerator
            self.model = GPTImageGenerator(
                model="gpt-image-1",
                app_id="appduby53au4sc7wisn",
                app_key="JFXqxpTAGIZdyVzLudJJikPsyHSReoSf"
            )
        elif model_name == "nextstep1":
            # from models.NextStep1.modeling_nextstep1 import NextStep1
            # self.model = NextStep1(model_path="/mnt/shenzhen2cephfs/mm-base-vision/kotisye/pretrain/NextStep-1-Large-Edit")
            self.model = "NextStep1 model placeholder"
        else:
            raise ValueError(f"Invalid model name: {model_name}")
        logging.info(f"Initialized EditModel: {model_name}")
    
    def process(self, image: Image.Image, instruction: str) -> Image.Image:
        """
        Process image with editing instruction
        
        Args:
            image: Input PIL Image
            instruction: Editing instruction text
            
        Returns:
            Edited PIL Image
        """
        edited_image = self.model.process(image, instruction)
        return edited_image


class VLMModel:
    """Base class for VLM evaluation models"""
    def __init__(self, model_name: str = "gpt-4.1", model_path: Optional[str] = None):
        """
        Initialize VLM model
        
        Args:
            model_name: Name of the VLM model
            model_path: Path to model weights (for local models)
        """
        self.model_name = model_name
        self.model_path = model_path

        if model_name == "gpt-4.1":
            from models.ichat_gpt4o import Ichat_GPT4o
            self.model = Ichat_GPT4o(model="gpt-4.1")
        elif model_name == "qwen25vl":
            from models.qwen_vl_base import QwenVL
            self.model = QwenVL(model_path=model_path, tensor_parallel_size=8)
        
        logging.info(f"Initialized VLMModel: {model_name}")
    
    def prepare_prompt(self, text_prompt: str, images: List[Image.Image]) -> Any:
        """
        Prepare prompt with text and images
        
        Args:
            text_prompt: Text prompt string
            images: List of PIL Images
            
        Returns:
            Prepared prompt for the model
        """
        return self.model.prepare_prompt(images, text_prompt)
    
    def forward(self, final_prompt: Any) -> str:
        """
        Forward pass through VLM model
        
        Args:
            final_prompt: Prepared prompt containing text and images
            
        Returns:
            Model output string
        """
        return self.model.forward(final_prompt)


class WeBenchmark:
    """Main benchmark class for image editing evaluation"""
    
    def __init__(
        self,
        data_path: str,
        save_dir: str,
        edit_model: Optional[EditModel] = None,
        edit_model_name: str = "qwen-image-edit",
        vlm_model: Optional[VLMModel] = None,
        languages: List[str] = ["en"],
        num_workers: int = 1,
        single: bool = False,
        # edit_workers: int = 1,
        skip_editing: bool = False,
        skip_evaluation: bool = False
    ):
        """
        Initialize benchmark
        
        Args:
            data_path: Path to test data jsonl file
            save_dir: Root directory to save results
            edit_model: Image editing model instance
            vlm_model: VLM evaluation model instance
            languages: List of languages to test ["en", "cn", or both]
            num_workers: Number of workers for parallel evaluation (only for GPT models)
            skip_editing: If True, skip editing and only run evaluation
        """
        self.data_path = data_path
        self.save_dir = save_dir
        self.edit_model = edit_model
        self.edit_model_name = edit_model_name
        self.vlm_model = vlm_model
        self.languages = languages
        self.num_workers = num_workers
        self.single = single
        # self.edit_workers = edit_workers
        self.skip_editing = skip_editing
        self.skip_evaluation = skip_evaluation
        
        # Load evaluation prompts
        self.if_prompt = prompts._prompts_if
        self.nc_prompt = prompts._prompts_nc
        self.vq_prompt = prompts._prompts_vq
        self.ra_prompt = prompts._prompts_ra

        # Load test data
        self.data_list = self._load_data()
        # self.language = self._load_language()
        
        logging.info(f"Loaded {len(self.data_list)} test samples")
        logging.info(f"Languages: {languages}")
        logging.info(f"Skip editing: {skip_editing}")
        logging.info(f"Number of workers: {num_workers}")
    
    def _load_data(self) -> List[Dict]:
        """Load test data from jsonl file"""
        data_list = []
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line in f:
                data_list.append(json.loads(line.strip()))
        return data_list
    
    # def _load_language(self) -> str:
    #     if isinstance(self.languages, str) and self.languages in ["en", "cn"]:
    #         return [self.languages]
    #     elif isinstance(self.languages, str) and self.languages == "all":
    #         return ["en", "cn"]
    #     elif isinstance(self.languages, list) and all(lang in ["en", "cn"] for lang in self.languages):
    #         return self.languages
    #     else:
    #         raise ValueError(f"Invalid languages: {self.languages}")
    
    def _organize_data_by_task_subtask(self) -> Dict[str, Dict[str, List[Dict]]]:
        """
        Organize data by task and subtask
        
        Returns:
            Nested dictionary: {task: {subtask: [samples]}}
        """
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
    
    def _is_api_based_model(self) -> bool:
        """
        Check if the editing model is API-based (supports multi-threading)
        
        Returns:
            True if model is API-based, False otherwise
        """
        api_based_models = ["nano_banana", "gpt-4o", "gpt-image-1"]
        return self.edit_model_name in api_based_models
    
    def _edit_single_sample(
        self,
        sample: Dict,
        language: str,
        img_dir: str
    ) -> Tuple[bool, str]:
        """
        Edit a single sample
        
        Args:
            sample: Data sample dictionary
            language: Language code
            img_dir: Directory to save edited image
            
        Returns:
            Tuple of (success, error_message)
        """
        try:
            # Get instruction
            instruction = sample[language]
            key = sample["key"]
            
            # Load original image
            original_image = Image.open(sample["image_path"]).convert("RGB")
            
            # Edit image
            img_path = os.path.join(img_dir, f"{key}.png")
            if os.path.exists(img_path):
                return True, ""
            edited_image = self.edit_model.process(original_image, instruction)
            
            # Save edited image
            edited_image.save(img_path)
            
            return True, ""
            
        except Exception as e:
            error_msg = f"Error editing sample {sample['key']}: {e}"
            logging.error(error_msg)
            return False, error_msg
    
    def _edit_single_wrapper(self, args: Tuple) -> Tuple[bool, str]:
        """
        Wrapper for single sample editing (for parallel processing)
        
        Args:
            args: Tuple of (sample, language, img_dir)
            
        Returns:
            Tuple of (success, error_message)
        """
        sample, language, img_dir = args
        return self._edit_single_sample(sample, language, img_dir)


    
    def edit(self) -> None:
        """Run image editing on all test samples"""
        if self.skip_editing:
            logging.info("Skipping editing phase as requested")
            return
        
        if self.edit_model is None:
            logging.error("Edit model is not provided. Cannot run editing.")
            return
        
        logging.info("="*60)
        logging.info("Starting Image Editing Phase")
        logging.info("="*60)
        
        # Check if model supports multi-threading
        use_multithreading = self._is_api_based_model() and self.num_workers > 1 and not self.single
        if use_multithreading:
            logging.info(f"Using multi-threading with {self.num_workers} workers for API-based model")
        else:
            logging.info("Using single-threaded processing")
        
        # Organize data by task and subtask
        organized_data = self._organize_data_by_task_subtask()
        model_name = self.edit_model_name
        
        for language in self.languages:
            logging.info(f"{'='*60}")
            logging.info(f"Processing Language: {language.upper()}")
            logging.info(f"{'='*60}")
            
            # Process by task and subtask
            for task_idx, (task, subtasks) in enumerate(organized_data.items(), 1):
                logging.info(f"\n[Task {task_idx}/{len(organized_data)}] {task}")
                
                for subtask_idx, (subtask, samples) in enumerate(subtasks.items(), 1):
                    logging.info(f"  [Subtask {subtask_idx}/{len(subtasks)}] {subtask} - {len(samples)} samples")
                    
                    # Create save directory for this subtask
                    subtask_name = '_'.join(subtask.split())
                    img_dir = os.path.join(self.save_dir, model_name, subtask_name, language)
                    os.makedirs(img_dir, exist_ok=True)
                    
                    # Prepare editing tasks
                    edit_tasks = [(sample, language, img_dir) for sample in samples]
                    
                    # Process samples with multi-threading or single-threading
                    if use_multithreading:
                        # Multi-threaded editing for API-based models
                        success_count = 0
                        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                            futures = [executor.submit(self._edit_single_wrapper, task) for task in edit_tasks]
                            
                            for future in tqdm(as_completed(futures), total=len(futures),
                                             desc=f"  Editing {task}/{subtask}", leave=False):
                                success, error_msg = future.result()
                                if success:
                                    success_count += 1
                        
                        logging.info(f"  Successfully edited {success_count}/{len(samples)} samples")
                    else:
                        # Single-threaded editing for local models
                        success_count = 0
                        for sample in tqdm(samples, desc=f"  Editing {task}/{subtask}", leave=False):
                            success, error_msg = self._edit_single_sample(sample, language, img_dir)
                            if success:
                                success_count += 1
                        
                        logging.info(f"  Successfully edited {success_count}/{len(samples)} samples")
        
        logging.info("\n" + "="*60)
        logging.info("Image editing completed!")
        logging.info("="*60)


    
    def _parse_vlm_response(self, response: str) -> Dict[str, Any]:
        """
        Parse VLM response to extract score and reason
        
        Args:
            response: Raw response string from VLM
            
        Returns:
            Dictionary with 'score' and 'reason' keys
        """
        # Try to parse as JSON
        response = response.replace('```json', '').replace('```', '').strip()

        # Convert Python tuples to lists in the string
        response = response.replace('(', '[').replace(')', ']')
        try:
            result = json.loads(response)
            return {
                "score": float(result.get("score", 0)),
                "reason": result.get("reason", "")
            }
        except:
            # If parsing fails, return default values
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
        """
        Evaluate a single sample with all metrics
        
        Args:
            sample: Data sample dictionary
            language: Language code
            original_image: Original PIL Image
            edited_image: Edited PIL Image
            edited_image_path: Path to edited image
            
        Returns:
            Dictionary containing all evaluation results
        """
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

        
        # Evaluate IF (Instruction Following)
        if_prompt_text = self.if_prompt.replace("<instruction>", instruction)
        if_final_prompt = self.vlm_model.prepare_prompt(if_prompt_text, images)
        if_response = self.vlm_model.forward(if_final_prompt)
        if_result = self._parse_vlm_response(if_response)
        result["if_score"] = if_result["score"]
        result["if_reason"] = if_result["reason"]
        
        # Evaluate NC (Non-edited region Consistency)
        nc_prompt_text = self.nc_prompt.replace("<instruction>", instruction)
        nc_final_prompt = self.vlm_model.prepare_prompt(nc_prompt_text, images)
        nc_response = self.vlm_model.forward(nc_final_prompt)
        nc_result = self._parse_vlm_response(nc_response)
        result["nc_score"] = nc_result["score"]
        result["nc_reason"] = nc_result["reason"]
        
        # Evaluate VQ (Visual Quality)
        vq_prompt_text = self.vq_prompt.replace("<instruction>", instruction)
        vq_final_prompt = self.vlm_model.prepare_prompt(vq_prompt_text, images)
        vq_response = self.vlm_model.forward(vq_final_prompt)
        vq_result = self._parse_vlm_response(vq_response)
        result["vq_score"] = vq_result["score"]
        result["vq_reason"] = vq_result["reason"]
        
        # Evaluate RA (Reasoning Accuracy) - only for Reasoning Editing tasks
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
            
            # Calculate overall score with 4 metrics
            scores = [if_result["score"], nc_result["score"], vq_result["score"], ra_result["score"]]
            result["overall_score"] = np.power(np.prod(scores), 1.0/4.0)
        else:
            result["ra_score"] = None
            result["ra_reason"] = None
            
            # Calculate overall score with 3 metrics
            scores = [if_result["score"], nc_result["score"], vq_result["score"]]
            result["overall_score"] = np.power(np.prod(scores), 1.0/3.0)
        
        return result
    
    def _evaluate_single_wrapper(self, args: Tuple) -> Dict[str, Any]:
        """
        Wrapper for single sample evaluation (for parallel processing)
        
        Args:
            args: Tuple of (sample, language, original_image, edited_image, edited_image_path)
            
        Returns:
            Evaluation result dictionary
        """
        sample, language, original_image, edited_image, edited_image_path = args
        try:
            return self._evaluate_single_sample(sample, language, original_image, edited_image, edited_image_path)
        except Exception as e:
            logging.error(f"Error evaluating sample {sample['key']}: {e}")
            return None

    
    def eval(self) -> None:
        """Run evaluation on all edited images"""
        if self.vlm_model is None:
            logging.error("VLM model is not provided. Cannot run evaluation.")
            return
        
        logging.info("="*60)
        logging.info("Starting Evaluation Phase")
        logging.info("="*60)
        
        # Organize data by task and subtask
        organized_data = self._organize_data_by_task_subtask()
        model_name = self.edit_model_name if self.edit_model_name else "unknown_model"
        vlm_name = self.vlm_model.model_name
        
        for language in self.languages:
            logging.info(f"{'='*60}")
            logging.info(f"Evaluating Language: {language.upper()}")
            logging.info(f"{'='*60}")
            
            # Process by task and subtask
            for task_idx, (task, subtasks) in enumerate(organized_data.items(), 1):
                logging.info(f"\n[Task {task_idx}/{len(organized_data)}] {task}")
                
                for subtask_idx, (subtask, samples) in enumerate(subtasks.items(), 1):
                    logging.info(f"  [Subtask {subtask_idx}/{len(subtasks)}] {subtask} - {len(samples)} samples")
                    
                    # Prepare paths for this subtask
                    subtask_name = '_'.join(subtask.split())
                    img_dir = os.path.join(self.save_dir, model_name, subtask_name, language)
                    eval_dir = os.path.join(self.save_dir, model_name, "eval_output", vlm_name)
                    os.makedirs(eval_dir, exist_ok=True)
                    eval_file = os.path.join(eval_dir, f"{subtask_name}_{language}_results.jsonl")

                    # Check whether this subtask has already been evaluated
                    if os.path.exists(eval_file):
                        logging.info(f"  Skipping {subtask} as it has already been evaluated")
                        continue
                    
                    # Prepare evaluation tasks for this subtask
                    eval_tasks = []
                    for sample in samples:
                        try:
                            # Load images
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
                    
                    # Run evaluation for this subtask
                    results = []
                    
                    # Use multi-threading for GPT models
                    if "gpt" in vlm_name.lower() and self.num_workers > 1:
                        logging.info(f"Using multi-threading with {self.num_workers} workers for Evaluation")
                        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                            futures = [executor.submit(self._evaluate_single_wrapper, task) for task in eval_tasks]
                            
                            for future in tqdm(as_completed(futures), total=len(futures), 
                                             desc=f"  Evaluating {task}/{subtask}", leave=False):
                                result = future.result()
                                if result is not None:
                                    results.append(result)
                    else:
                        # Single-threaded evaluation
                        logging.info("Using single-threaded evaluation")
                        for task_data in tqdm(eval_tasks, desc=f"  Evaluating {task}/{subtask}", leave=False):
                            result = self._evaluate_single_wrapper(task_data)
                            if result is not None:
                                results.append(result)
                    
                    # Save results for this subtask
                    if results:
                        with open(eval_file, 'w', encoding='utf-8') as f:
                            for item in results:
                                f.write(json.dumps(item, ensure_ascii=False) + '\n')
                        
                        logging.info(f"  Saved {len(results)} results to {eval_file}")
        
        logging.info("\n" + "="*60)
        logging.info("Evaluation completed!")
        logging.info("="*60)

    
    def compute_statistics(self) -> Dict[str, Any]:
        """
        Compute statistics for all evaluation results
        
        Returns:
            Dictionary containing statistics at different levels
        """
        logging.info("="*60)
        logging.info("Computing Statistics")
        logging.info("="*60)
        
        model_name = self.edit_model_name if self.edit_model_name else "unknown_model"
        vlm_name = self.vlm_model.model_name if self.vlm_model else "unknown_vlm"
        
        all_statistics = {}
        
        for language in self.languages:
            logging.info(f"\nComputing statistics for language: {language}")
            
            # Collect all results
            all_results = []
            eval_base_dir = os.path.join(
                self.save_dir,
                model_name,
                "eval_output",
                vlm_name
            )
            
            # Find all result files
            for subtask_filer in os.listdir(eval_base_dir):
                if subtask_filer.endswith(f"{language}_results.jsonl"):
                    subtask_path = os.path.join(eval_base_dir, subtask_filer)
                    if os.path.exists(subtask_path):
                        with open(subtask_path, 'r', encoding='utf-8') as f:
                            for line in f:
                                all_results.append(json.loads(line.strip()))
                else:
                    continue
            
            if not all_results:
                logging.warning(f"No results found for language: {language}")
                continue
            
            # Compute overall statistics
            overall_stats = self._compute_stats_for_group(all_results, "Overall")
            
            # Compute task-level statistics
            task_stats = {}
            tasks = set([r["task"] for r in all_results])
            for task in tasks:
                task_results = [r for r in all_results if r["task"] == task]
                task_stats[task] = self._compute_stats_for_group(task_results, task)
            
            # Compute subtask-level statistics
            subtask_stats = {}
            subtasks = set([r["subtask"] for r in all_results])
            for subtask in subtasks:
                subtask_results = [r for r in all_results if r["subtask"] == subtask]
                subtask_stats[subtask] = self._compute_stats_for_group(subtask_results, subtask)
            
            # Store statistics
            all_statistics[language] = {
                "overall": overall_stats,
                "by_task": task_stats,
                "by_subtask": subtask_stats
            }
            
            # Save statistics to file
            stats_dir = os.path.join(
                self.save_dir,
                model_name,
                "eval_output",
                vlm_name,
                "statistics"
            )
            os.makedirs(stats_dir, exist_ok=True)
            
            stats_file = os.path.join(stats_dir, f"{language}_statistics.json")
            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump(all_statistics[language], f, indent=2, ensure_ascii=False)
            
            logging.info(f"Saved statistics to {stats_file}")
            
            # Print summary
            self._print_statistics_summary(all_statistics[language], language)
        
        return all_statistics
    
    def _compute_stats_for_group(self, results: List[Dict], group_name: str) -> Dict[str, float]:
        """
        Compute statistics for a group of results
        
        Args:
            results: List of result dictionaries
            group_name: Name of the group
            
        Returns:
            Dictionary containing mean scores
        """
        if not results:
            return {}
        
        stats = {
            "count": len(results),
            "if_mean": np.mean([r["if_score"] for r in results]),
            "nc_mean": np.mean([r["nc_score"] for r in results]),
            "vq_mean": np.mean([r["vq_score"] for r in results]),
            "overall_mean": np.mean([r["overall_score"] for r in results])
        }
        
        # Add RA statistics if applicable
        ra_scores = [r["ra_score"] for r in results if r["ra_score"] is not None]
        if ra_scores:
            stats["ra_mean"] = np.mean(ra_scores)
            stats["ra_count"] = len(ra_scores)
        
        return stats
    
    def _print_statistics_summary(self, stats: Dict, language: str) -> None:
        """Print statistics summary"""
        logging.info(f"\n{'='*60}")
        logging.info(f"Statistics Summary - {language.upper()}")
        logging.info(f"{'='*60}")
        
        # Overall statistics
        overall = stats["overall"]
        logging.info(f"\nOverall (n={overall['count']}):")
        logging.info(f"  IF:      {overall['if_mean']:.4f}")
        logging.info(f"  NC:      {overall['nc_mean']:.4f}")
        logging.info(f"  VQ:      {overall['vq_mean']:.4f}")
        if "ra_mean" in overall:
            logging.info(f"  RA:      {overall['ra_mean']:.4f} (n={overall['ra_count']})")
        logging.info(f"  Overall: {overall['overall_mean']:.4f}")
        
        # Task-level statistics
        logging.info(f"\nBy Task:")
        for task, task_stats in stats["by_task"].items():
            logging.info(f"  {task} (n={task_stats['count']}):")
            logging.info(f"    IF: {task_stats['if_mean']:.4f}, NC: {task_stats['nc_mean']:.4f}, "
                        f"VQ: {task_stats['vq_mean']:.4f}, Overall: {task_stats['overall_mean']:.4f}")
            if "ra_mean" in task_stats:
                logging.info(f"    RA: {task_stats['ra_mean']:.4f}")
        
        # Subtask-level statistics
        logging.info(f"\nBy Subtask:")
        for subtask, subtask_stats in stats["by_subtask"].items():
            logging.info(f"  {subtask} (n={subtask_stats['count']}):")
            logging.info(f"    IF: {subtask_stats['if_mean']:.4f}, NC: {subtask_stats['nc_mean']:.4f}, "
                        f"VQ: {subtask_stats['vq_mean']:.4f}, Overall: {subtask_stats['overall_mean']:.4f}")
            if "ra_mean" in subtask_stats:
                logging.info(f"    RA: {subtask_stats['ra_mean']:.4f}")
    
    def run(self) -> None:
        """Run complete benchmark pipeline"""
        logging.info("="*60)
        logging.info("Starting WeBenchmark Pipeline")
        logging.info("="*60)
        
        start_time = time.time()
        
        # Step 1: Edit images
        if not self.skip_editing:
            self.edit()
        
        # Step 2: Evaluate
        if not self.skip_evaluation:
            self.eval()
        
        # Step 3: Compute statistics
        self.compute_statistics()
        
        elapsed_time = time.time() - start_time
        logging.info(f"\n{'='*60}")
        logging.info(f"Pipeline completed in {elapsed_time:.2f} seconds")
        logging.info(f"{'='*60}")


def setup_logging(log_level: str = "INFO") -> None:
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('benchmark.log')
        ]
    )


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Image Editing Benchmark Pipeline")
    
    # Data arguments
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to test data jsonl file')
    parser.add_argument('--save_dir', type=str, required=True,
                       help='Root directory to save results')
    
    # Model arguments
    parser.add_argument('--edit_model_name', type=str, default='default_model',
                       help='Name of the editing model')
    parser.add_argument('--vlm_model_name', type=str, default='gpt-4.1',
                       help='Name of the VLM evaluation model')
    parser.add_argument('--vlm_model_path', type=str, default=None,
                       help='Path to VLM model weights (for local models)')
    
    # Language arguments
    parser.add_argument('--languages', type=str, nargs='+', default=['en'],
                       choices=['en', 'cn'],
                       help='Languages to test (en, cn, or both)')
    
    # Execution arguments
    parser.add_argument('--skip_editing', action='store_true',
                       help='Skip editing phase and only run evaluation')
    parser.add_argument('--skip_evaluation', action='store_true',
                        help='Skip evaluation phase and only run editing')
    parser.add_argument('--num_workers', type=int, default=1,
                       help='Number of workers for parallel evaluation (only for GPT models)')
    parser.add_argument('--single', action='store_true', help='Run single worker')
    
    # Logging arguments
    parser.add_argument('--log_level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    
    # Seed argument
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    return parser.parse_args()


def main():
    """Main function"""
    args = parse_arguments()
    
    # Setup logging
    setup_logging(args.log_level)

    # Set seed
    set_seed(args.seed)
    
    # Initialize models (users should implement their own model classes)
    if not args.skip_editing:
        edit_model = EditModel(model_name=args.edit_model_name)
    vlm_model = VLMModel(model_name=args.vlm_model_name, model_path=args.vlm_model_path)
    
    # Initialize benchmark
    benchmark = WeBenchmark(
        data_path=args.data_path,
        save_dir=args.save_dir,
        edit_model=edit_model if not args.skip_editing else None,
        edit_model_name=args.edit_model_name,
        vlm_model=vlm_model,
        languages=args.languages,
        num_workers=args.num_workers,
        single=args.single,
        skip_editing=args.skip_editing,
        skip_evaluation=args.skip_evaluation
    )
    
    # Run pipeline
    benchmark.run()


if __name__ == "__main__":
    main()