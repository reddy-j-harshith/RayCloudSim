#!/usr/bin/env python3
"""Quick All Datasets Evaluation
=================================

Research-grade quick evaluation with comprehensive metrics and visualizations.
"""

import os
import json
import math
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from research_attack_aware_system import ResearchAttackAwareSystem

sns.set_theme(style="darkgrid")


class QuickAllDatasetsEvaluator:
    """Quick evaluation for all datasets with comprehensive analytics."""

    def __init__(self,
                 malicious_ratio: float = 0.30,
                 training_epochs: int = 15,  # Reduced from 25
                 training_cycles: int = 8,   # Reduced from 12
                 testing_cycles: int = 6,    # Reduced from 8
                 trust_cycles: int = 6,      # Reduced from 10
                 detection_threshold: float = 0.45):
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.results_dir = f"midsem_results/quick_all_datasets_{self.timestamp}"
        os.makedirs(self.results_dir, exist_ok=True)

        self.malicious_ratio = malicious_ratio
        self.training_epochs = training_epochs
        self.training_cycles = training_cycles
        self.testing_cycles = testing_cycles
        self.trust_cycles = trust_cycles
        self.detection_threshold = detection_threshold

        self.datasets = {
            "pakistan": ["Tuple30K", "Tuple50K", "Tuple100K"],
            "topo4mec": ["25N50E", "50N50E", "100N150E", "MilanCityCenter"]
        }

        self.dataset_metrics: Dict[str, Dict[str, Any]] = {}

        print("🚀 Quick All Datasets Evaluator (Research Grade)")
        print(f"📁 Results directory: {self.results_dir}")
        print(f"🔧 Malicious ratio: {malicious_ratio * 100:.0f}%")
        print(f"📊 Total datasets: {sum(len(v) for v in self.datasets.values())}")
        print("✨ Outputs: Trust trajectories, loss curves, attack timelines, detection metrics, network protection, precision/recall/F1")

    # ---------------------------------------------------------------------
    # Top-level coordination
    # ---------------------------------------------------------------------
    def run_all_datasets(self) -> Dict[str, Dict[str, Any]]:
        print("\n🚀 Beginning quick comprehensive evaluation across all datasets...\n")
        all_results: Dict[str, Dict[str, Any]] = {}

        total = sum(len(s) for s in self.datasets.values())
        processed = 0

        for dataset_type, subsets in self.datasets.items():
            for subset_name in subsets:
                processed += 1
                dataset_key = f"{dataset_type}_{subset_name}"
                print("=" * 90)
                print(f"DATASET {processed}/{total}: {dataset_type.upper()} - {subset_name}")
                print("=" * 90)

                try:
                    result_summary = self._process_single_dataset(dataset_type, subset_name)
                    all_results[dataset_key] = result_summary
                except Exception as exc:  # pragma: no cover - defensive logging
                    print(f"❌ Failure while processing {dataset_key}: {exc}")
                    all_results[dataset_key] = {"error": str(exc)}

        # Persist aggregated results
        results_path = os.path.join(self.results_dir, "all_datasets_results.json")
        with open(results_path, "w", encoding="utf-8") as fh:
            json.dump(all_results, fh, indent=2, default=str)

        self._generate_summary(all_results)
        self._generate_html_report()

        print("\n🎉 Quick comprehensive evaluation complete!")
        print(f"📁 Detailed outputs: {self.results_dir}")
        print(f"📄 Aggregated JSON: {results_path}")

        return all_results

    def _process_single_dataset(self, dataset_type: str, subset_name: str) -> Dict[str, Any]:
        dataset_key = f"{dataset_type}_{subset_name}"
        dataset_dir = os.path.join(self.results_dir, dataset_key)
        os.makedirs(dataset_dir, exist_ok=True)

        phase_dirs = {
            "training": os.path.join(dataset_dir, "training"),
            "testing": os.path.join(dataset_dir, "testing"),
            "trust": os.path.join(dataset_dir, "trust_based"),
            "baseline": os.path.join(dataset_dir, "baseline")
        }
        for directory in phase_dirs.values():
            os.makedirs(directory, exist_ok=True)

        system = ResearchAttackAwareSystem(malicious_ratio=self.malicious_ratio, output_dir=phase_dirs["training"])

        # Training phase
        print("   🏋️ Training phase with loss tracking ...")
        system.output_dir = phase_dirs["training"]
        training_results = system.run_comprehensive_attack_simulation(
            dataset_name=dataset_type,
            dataset_flag=subset_name,
            output_dir=phase_dirs["training"],
            model_type="gat",
            malicious_ratio=self.malicious_ratio,
            num_epochs=self.training_epochs,
            task_cycles=self.training_cycles,
            save_models=False,
            test_mode=False,
            enable_trust_offloading=False
        )

        # Testing baseline (no offloading)
        print("   🔍 Testing phase (baseline inference) ...")
        system.output_dir = phase_dirs["testing"]
        testing_results = system.run_comprehensive_attack_simulation(
            dataset_name=dataset_type,
            dataset_flag=subset_name,
            output_dir=phase_dirs["testing"],
            model_type="gat",
            malicious_ratio=self.malicious_ratio,
            num_epochs=0,
            task_cycles=self.testing_cycles,
            save_models=False,
            test_mode=True,
            enable_trust_offloading=False
        )

        # Trust-based offloading
        print("   🛡️ Trust-based offloading evaluation ...")
        system.output_dir = phase_dirs["trust"]
        trust_results = system.run_comprehensive_attack_simulation(
            dataset_name=dataset_type,
            dataset_flag=subset_name,
            output_dir=phase_dirs["trust"],
            model_type="gat",
            malicious_ratio=self.malicious_ratio,
            num_epochs=0,
            task_cycles=self.trust_cycles,
            save_models=False,
            test_mode=True,
            enable_trust_offloading=True
        )

        # Baseline offloading
        print("   🧪 Baseline offloading benchmark ...")
        system.output_dir = phase_dirs["baseline"]
        baseline_results = system.run_comprehensive_attack_simulation(
            dataset_name=dataset_type,
            dataset_flag=subset_name,
            output_dir=phase_dirs["baseline"],
            model_type="gat",
            malicious_ratio=self.malicious_ratio,
            num_epochs=0,
            task_cycles=self.trust_cycles,
            save_models=False,
            test_mode=True,
            enable_trust_offloading=False
        )

        # Build comprehensive metrics & plots
        metrics = self._build_comprehensive_metrics(
            dataset_key,
            training_results,
            testing_results,
            trust_results,
            baseline_results
        )
        self.dataset_metrics[dataset_key] = metrics

        self._save_metrics(dataset_dir, metrics)
        self._create_visualizations(dataset_key, metrics, dataset_dir)

        summary = {
            "dataset": dataset_key,
            "training_success_rate": metrics["phases"]["training"]["success_rate"],
            "testing_success_rate": metrics["phases"]["testing"]["success_rate"],
            "trust_success_rate": metrics["phases"]["trust_based"]["success_rate"],
            "baseline_success_rate": metrics["phases"]["baseline"]["success_rate"],
            "trust_improvement": metrics["improvements"]["success_rate_improvement"],
            "precision": metrics["classification_metrics"]["precision"],
            "recall": metrics["classification_metrics"]["recall"],
            "f1_score": metrics["classification_metrics"]["f1_score"],
            "accuracy": metrics["classification_metrics"]["accuracy"]
        }

        print(
            f"   ✅ Completed {dataset_key}: test success {summary['testing_success_rate']:.2%}, "
            f"trust-based {summary['trust_success_rate']:.2%}, Δ {summary['trust_improvement']:.3f}"
        )

        return summary

    # ------------------------------------------------------------------
    # Metrics extraction helpers
    # ------------------------------------------------------------------
    def _build_comprehensive_metrics(self,
                                     dataset_name: str,
                                     training_results: Dict[str, Any],
                                     testing_results: Dict[str, Any],
                                     trust_results: Dict[str, Any],
                                     baseline_results: Dict[str, Any]) -> Dict[str, Any]:
        phases = {
            "training": self._extract_phase_metrics(training_results, "Training"),
            "testing": self._extract_phase_metrics(testing_results, "Testing"),
            "trust_based": self._extract_phase_metrics(trust_results, "Trust-Based"),
            "baseline": self._extract_phase_metrics(baseline_results, "Baseline")
        }

        trust_trajectories = self._extract_trust_trajectories(trust_results)
        loss_curves = self._extract_loss_curves(training_results)
        attack_logs = self._extract_attack_logs(trust_results, baseline_results)
        classification_metrics = self._calculate_classification_metrics(trust_results)
        protection_metrics = self._calculate_protection_metrics(trust_results, baseline_results)

        baseline_sr = phases["baseline"]["success_rate"] or 1e-9
        improvements = {
            "success_rate_improvement": phases["trust_based"]["success_rate"] - phases["baseline"]["success_rate"],
            "latency_improvement": phases["baseline"]["avg_latency"] - phases["trust_based"]["avg_latency"],
            "energy_improvement": phases["baseline"]["energy_consumption"] - phases["trust_based"]["energy_consumption"],
            "improvement_percentage": (
                (phases["trust_based"]["success_rate"] - phases["baseline"]["success_rate"]) / baseline_sr * 100.0
            )
        }

        malicious_nodes = trust_results.get("malicious_nodes", [])
        all_nodes = trust_results.get("honest_nodes", []) + malicious_nodes
        network_info = {
            "total_nodes": len(all_nodes),
            "malicious_nodes": len(malicious_nodes),
            "honest_nodes": len(all_nodes) - len(malicious_nodes)
        }

        trust_analysis = self._calculate_trust_analysis(trust_results)

        model_performance = {
            "train_accuracy": training_results.get("final_train_accuracy", 0.0),
            "val_accuracy": training_results.get("final_val_accuracy", 0.0),
            "train_rmse": training_results.get("train_rmse", 0.0),
            "val_rmse": training_results.get("val_rmse", 0.0),
            "test_accuracy": training_results.get("test_accuracy", 0.0),
            "test_precision": training_results.get("test_precision", 0.0),
            "test_recall": training_results.get("test_recall", 0.0),
            "test_f1": training_results.get("test_f1", 0.0)
        }

        return {
            "dataset": dataset_name,
            "phases": phases,
            "trust_trajectories": trust_trajectories,
            "loss_curves": loss_curves,
            "attack_logs": attack_logs,
            "classification_metrics": classification_metrics,
            "protection_metrics": protection_metrics,
            "improvements": improvements,
            "network_info": network_info,
            "trust_analysis": trust_analysis,
            "model_performance": model_performance
        }

    def _extract_phase_metrics(self, results: Dict[str, Any], phase_name: str) -> Dict[str, Any]:
        execution = results.get("execution_results", {})
        logger = execution.get("phase_logger")
        task_logs = getattr(logger, "task_logs", []) or []

        successful = execution.get("successful_tasks", 0)
        failed = execution.get("failed_tasks", 0)
        total = successful + failed
        success_rate = successful / total if total else 0.0

        if task_logs:
            log_df = pd.DataFrame(task_logs)
            latencies = log_df.get("execution_time", pd.Series(dtype=float)).to_numpy()
            energies = log_df.get("energy_consumed", pd.Series(dtype=float)).to_numpy()
        else:
            latencies = np.array([])
            energies = np.array([])

        return {
            "phase_name": phase_name,
            "total_tasks": total,
            "successful_tasks": successful,
            "failed_tasks": failed,
            "success_rate": success_rate,
            "avg_latency": float(np.mean(latencies)) if latencies.size else 0.0,
            "median_latency": float(np.median(latencies)) if latencies.size else 0.0,
            "p95_latency": float(np.percentile(latencies, 95)) if latencies.size else 0.0,
            "avg_energy": float(np.mean(energies)) if energies.size else 0.0,
            "energy_consumption": float(np.sum(energies)) if energies.size else 0.0
        }

    def _extract_trust_trajectories(self, results: Dict[str, Any]) -> Dict[str, Any]:
        execution = results.get("execution_results", {})
        temporal = execution.get("temporal_trust_data", []) or []
        logger = execution.get("phase_logger")
        task_logs = getattr(logger, "task_logs", []) or []

        if not temporal:
            return {
                "timeline": [],
                "honest_trust": [],
                "malicious_trust": [],
                "trust_gap_over_time": [],
                "attacks": []
            }

        temporal_df = pd.DataFrame(temporal)
        timeline = sorted(temporal_df["task_index"].unique())
        honest_trust = []
        malicious_trust = []

        for idx in timeline:
            slice_df = temporal_df[temporal_df["task_index"] == idx]
            honest_vals = slice_df.loc[~slice_df["is_malicious"], "avg_trust"].values
            mal_vals = slice_df.loc[slice_df["is_malicious"], "avg_trust"].values
            honest_trust.append(float(np.mean(honest_vals)) if honest_vals.size else 0.6)
            malicious_trust.append(float(np.mean(mal_vals)) if mal_vals.size else 0.3)

        trust_gap = [h - m for h, m in zip(honest_trust, malicious_trust)]

        if timeline and task_logs:
            log_df = pd.DataFrame(task_logs)
            log_df["task_index"] = np.arange(len(log_df))
            window = timeline[1] - timeline[0] if len(timeline) > 1 else 100
            attacks = []
            for idx in timeline:
                mask = (log_df["task_index"] >= idx) & (log_df["task_index"] < idx + window)
                attacks.append(int(log_df.loc[mask, "is_dst_malicious"].sum()))
        else:
            attacks = [0] * len(timeline)

        return {
            "timeline": timeline,
            "honest_trust": honest_trust,
            "malicious_trust": malicious_trust,
            "trust_gap_over_time": trust_gap,
            "attacks": attacks
        }

    def _extract_loss_curves(self, results: Dict[str, Any]) -> Dict[str, Any]:
        train_losses = results.get("training_losses", []) or []
        val_losses = results.get("validation_losses", []) or []
        train_acc_curve = results.get("train_accuracy_curve", []) or []
        val_acc_curve = results.get("val_accuracy_curve", []) or []

        epochs = list(range(1, max(len(train_losses), len(val_losses), len(train_acc_curve), len(val_acc_curve)) + 1))
        loss_curves = {
            "gat": {
                "epochs": epochs,
                "train_loss": self._pad_curve(train_losses, len(epochs), fill_value=np.nan),
                "val_loss": self._pad_curve(val_losses, len(epochs), fill_value=np.nan),
                "train_acc": self._pad_curve(train_acc_curve, len(epochs), fill_value=np.nan),
                "val_acc": self._pad_curve(val_acc_curve, len(epochs), fill_value=np.nan)
            }
        }
        return loss_curves

    def _extract_attack_logs(self, trust_results: Dict[str, Any], baseline_results: Dict[str, Any]) -> Dict[str, Any]:
        trust_execution = trust_results.get("execution_results", {})
        baseline_execution = baseline_results.get("execution_results", {})

        trust_logger = trust_execution.get("phase_logger")
        baseline_logger = baseline_execution.get("phase_logger")

        trust_logs = getattr(trust_logger, "task_logs", []) or []
        baseline_logs = getattr(baseline_logger, "task_logs", []) or []

        trust_df = pd.DataFrame(trust_logs)
        baseline_df = pd.DataFrame(baseline_logs)

        def summarize(df: pd.DataFrame) -> List[Dict[str, Any]]:
            if df.empty:
                return []
            df = df.copy()
            df["task_index"] = np.arange(len(df))
            sample = df.head(250)  # cap for readability
            return sample[["task_index", "dst_node", "trust_score", "execution_success", "is_dst_malicious"]].to_dict("records")

        detection_times = self._compute_attack_detection_time(trust_df, trust_results.get("malicious_nodes", []))

        return {
            "trust_based_events": summarize(trust_df),
            "baseline_events": summarize(baseline_df),
            "attack_detection_time": detection_times,
            "total_attacks_trust": int(trust_df["is_dst_malicious"].sum()) if not trust_df.empty else 0,
            "total_attacks_baseline": int(baseline_df["is_dst_malicious"].sum()) if not baseline_df.empty else 0
        }

    def _calculate_classification_metrics(self, trust_results: Dict[str, Any]) -> Dict[str, Any]:
        execution = trust_results.get("execution_results", {})
        logger = execution.get("phase_logger")
        task_logs = getattr(logger, "task_logs", []) or []

        if not task_logs:
            return {
                "precision": 0.0,
                "recall": 0.0,
                "f1_score": 0.0,
                "accuracy": 0.0,
                "true_positives": 0,
                "false_positives": 0,
                "true_negatives": 0,
                "false_negatives": 0,
                "confusion_matrix": [[0, 0], [0, 0]]
            }

        df = pd.DataFrame(task_logs)
        df["y_true"] = df["is_dst_malicious"].astype(int)
        df["y_pred"] = (df["trust_score"] <= self.detection_threshold).astype(int)

        precision = precision_score(df["y_true"], df["y_pred"], zero_division=0)
        recall = recall_score(df["y_true"], df["y_pred"], zero_division=0)
        f1 = f1_score(df["y_true"], df["y_pred"], zero_division=0)
        accuracy = accuracy_score(df["y_true"], df["y_pred"])
        cm = confusion_matrix(df["y_true"], df["y_pred"])

        tn, fp, fn, tp = (cm.ravel().tolist() if cm.size == 4 else [0, 0, 0, 0])

        return {
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "accuracy": float(accuracy),
            "true_positives": int(tp),
            "false_positives": int(fp),
            "true_negatives": int(tn),
            "false_negatives": int(fn),
            "confusion_matrix": cm.tolist() if cm.size == 4 else [[tn, fp], [fn, tp]]
        }

    def _calculate_protection_metrics(self, trust_results: Dict[str, Any], baseline_results: Dict[str, Any]) -> Dict[str, Any]:
        trust_logs = pd.DataFrame(getattr(trust_results.get("execution_results", {}).get("phase_logger"), "task_logs", []))
        baseline_logs = pd.DataFrame(getattr(baseline_results.get("execution_results", {}).get("phase_logger"), "task_logs", []))

        def count_attacks(df: pd.DataFrame) -> Dict[str, int]:
            if df.empty:
                return {"total": 0, "successful": 0}
            total_attacks = int(df["is_dst_malicious"].sum())
            successful_attacks = int(df[(df["is_dst_malicious"] == True) & (df["execution_success"] == True)].shape[0])
            return {"total": total_attacks, "successful": successful_attacks}

        trust_counts = count_attacks(trust_logs)
        baseline_counts = count_attacks(baseline_logs)

        attacks_prevented = baseline_counts["successful"] - trust_counts["successful"]
        baseline_success = baseline_counts["successful"] or 1e-9
        prevention_rate = attacks_prevented / baseline_success

        return {
            "trust_total_attacks": trust_counts["total"],
            "baseline_total_attacks": baseline_counts["total"],
            "trust_successful_attacks": trust_counts["successful"],
            "baseline_successful_attacks": baseline_counts["successful"],
            "attacks_prevented": attacks_prevented,
            "prevention_rate": float(prevention_rate),
            "trust_blocked_attacks": trust_counts["total"] - trust_counts["successful"],
            "network_protection_improvement": float(prevention_rate * 100.0)
        }

    def _calculate_trust_analysis(self, trust_results: Dict[str, Any]) -> Dict[str, Any]:
        execution = trust_results.get("execution_results", {})
        final_matrix: Dict[int, Dict[int, float]] = execution.get("final_trust_matrix", {})
        malicious_nodes = trust_results.get("malicious_nodes", [])
        honest_nodes = trust_results.get("honest_nodes", [])

        malicious_values = []
        honest_values = []

        for src_node, trust_map in final_matrix.items():
            for dst_node, trust_value in trust_map.items():
                if dst_node in malicious_nodes:
                    malicious_values.append(trust_value)
                elif dst_node in honest_nodes:
                    honest_values.append(trust_value)

        mal_mean = float(np.mean(malicious_values)) if malicious_values else 0.3
        hon_mean = float(np.mean(honest_values)) if honest_values else 0.7
        trust_gap = hon_mean - mal_mean

        quality = "Excellent" if trust_gap > 0.3 else "Good" if trust_gap > 0.2 else "Fair" if trust_gap > 0.1 else "Needs Improvement"

        return {
            "malicious_trust_mean": mal_mean,
            "honest_trust_mean": hon_mean,
            "trust_gap": trust_gap,
            "separation_quality": quality
        }

    def _compute_attack_detection_time(self, trust_df: pd.DataFrame, malicious_nodes: List[int]) -> Dict[str, Any]:
        if trust_df.empty or not malicious_nodes:
            return {"avg_detection_delay_tasks": 0.0, "per_node": {}}

        trust_df = trust_df.copy()
        trust_df["task_index"] = np.arange(len(trust_df))
        temporal_map: Dict[int, List[int]] = defaultdict(list)
        per_node_delays = {}

        for node in malicious_nodes:
            node_attacks = trust_df[(trust_df["dst_node"] == node) & (trust_df["is_dst_malicious"].astype(bool))]
            if node_attacks.empty:
                continue

            first_attack_idx = int(node_attacks["task_index"].min())
            temporal_map[node].append(first_attack_idx)

        # We approximate detection as first time trust score dips below threshold after first attack
        for node, attack_indices in temporal_map.items():
            attack_idx = min(attack_indices)
            node_logs = trust_df[trust_df["dst_node"] == node]
            after_attack = node_logs[node_logs["task_index"] >= attack_idx]
            detected_rows = after_attack[after_attack["trust_score"] <= self.detection_threshold]
            if detected_rows.empty:
                delay = math.inf
            else:
                delay = float(detected_rows["task_index"].min() - attack_idx)
            per_node_delays[node] = delay

        finite_delays = [delay for delay in per_node_delays.values() if math.isfinite(delay)]
        average_delay = float(np.mean(finite_delays)) if finite_delays else float("inf")

        return {
            "avg_detection_delay_tasks": average_delay,
            "per_node": per_node_delays
        }

    # ------------------------------------------------------------------
    # Visualization helpers
    # ------------------------------------------------------------------
    def _create_visualizations(self, dataset_name: str, metrics: Dict[str, Any], output_dir: str) -> None:
        plots_dir = os.path.join(output_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)

        self._plot_trust_trajectories(metrics, plots_dir, dataset_name)
        self._plot_loss_curves(metrics, plots_dir, dataset_name)
        self._plot_attack_timeline(metrics, plots_dir, dataset_name)
        self._plot_classification_metrics(metrics, plots_dir, dataset_name)
        self._plot_protection_analysis(metrics, plots_dir, dataset_name)
        self._plot_performance_comparison(metrics, plots_dir, dataset_name)
        self._plot_trust_distribution(metrics, plots_dir, dataset_name)
        self._plot_confusion_matrix(metrics, plots_dir, dataset_name)

    def _plot_trust_trajectories(self, metrics: Dict[str, Any], plots_dir: str, dataset_name: str) -> None:
        trajectories = metrics["trust_trajectories"]
        if not trajectories["timeline"]:
            return

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
        fig.suptitle(f"{dataset_name} • Trust Trajectories During Attacks", fontweight="bold")

        ax1.plot(trajectories["timeline"], trajectories["honest_trust"], label="Honest", color="#2ecc71", marker="o", markersize=3)
        ax1.plot(trajectories["timeline"], trajectories["malicious_trust"], label="Malicious", color="#e74c3c", marker="x", markersize=3)
        ax1.axhline(self.detection_threshold, color="gray", linestyle="--", alpha=0.5, label="Detection Threshold")
        ax1.fill_between(trajectories["timeline"], trajectories["honest_trust"], trajectories["malicious_trust"], color="#3498db", alpha=0.2)
        ax1.set_ylabel("Average Trust")
        ax1.legend(loc="best")
        ax1.grid(alpha=0.3)

        ax2.plot(trajectories["timeline"], trajectories["trust_gap_over_time"], color="#9b59b6", linewidth=2, label="Trust Gap")
        ax2.bar(trajectories["timeline"], trajectories["attacks"], width=trajectory_window(trajectories["timeline"]), color="#e74c3c", alpha=0.4, label="Attack Count")
        ax2.set_xlabel("Task Index")
        ax2.set_ylabel("Trust Gap / Attack Count")
        ax2.legend(loc="upper right")
        ax2.grid(alpha=0.3)

        plt.tight_layout(rect=[0, 0, 1, 0.97])
        fig.savefig(os.path.join(plots_dir, f"{dataset_name}_trust_trajectories.png"), dpi=300)
        plt.close(fig)

    def _plot_loss_curves(self, metrics: Dict[str, Any], plots_dir: str, dataset_name: str) -> None:
        curves = metrics["loss_curves"]["gat"]
        if not curves["epochs"]:
            return

        fig, ax1 = plt.subplots(figsize=(14, 6))
        fig.suptitle(f"{dataset_name} • GAT Training Loss & Accuracy", fontweight="bold")

        ax1.plot(curves["epochs"], curves["train_loss"], color="#2980b9", marker="o", label="Train Loss")
        ax1.plot(curves["epochs"], curves["val_loss"], color="#c0392b", marker="s", label="Validation Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.grid(alpha=0.3)

        ax2 = ax1.twinx()
        ax2.plot(curves["epochs"], curves["train_acc"], color="#27ae60", linestyle="--", marker="^", label="Train Acc")
        ax2.plot(curves["epochs"], curves["val_acc"], color="#f1c40f", linestyle="--", marker="v", label="Val Acc")
        ax2.set_ylabel("Accuracy")

        lines_labels = ax1.get_legend_handles_labels()
        lines_labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_labels[0] + lines_labels2[0], lines_labels[1] + lines_labels2[1], loc="best")

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(os.path.join(plots_dir, f"{dataset_name}_loss_curves.png"), dpi=300)
        plt.close(fig)

    def _plot_attack_timeline(self, metrics: Dict[str, Any], plots_dir: str, dataset_name: str) -> None:
        attack_logs = metrics["attack_logs"]
        trust_events = pd.DataFrame(attack_logs["trust_based_events"])
        baseline_events = pd.DataFrame(attack_logs["baseline_events"])

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
        fig.suptitle(f"{dataset_name} • Attack Timeline", fontweight="bold")

        if not trust_events.empty:
            ax1.scatter(trust_events["task_index"], trust_events["is_dst_malicious"], c=trust_events["trust_score"], cmap="viridis", edgecolors="black", alpha=0.7)
        ax1.set_ylabel("Trust-Based (1=Malicious)")
        ax1.grid(alpha=0.3)

        if not baseline_events.empty:
            ax2.scatter(baseline_events["task_index"], baseline_events["is_dst_malicious"], c="#e74c3c", edgecolors="black", alpha=0.6)
        ax2.set_xlabel("Task Index")
        ax2.set_ylabel("Baseline (1=Malicious)")
        ax2.grid(alpha=0.3)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(os.path.join(plots_dir, f"{dataset_name}_attack_timeline.png"), dpi=300)
        plt.close(fig)

    def _plot_classification_metrics(self, metrics: Dict[str, Any], plots_dir: str, dataset_name: str) -> None:
        class_metrics = metrics["classification_metrics"]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(f"{dataset_name} • Detection Metrics", fontweight="bold")

        metric_names = ["Precision", "Recall", "F1-Score", "Accuracy"]
        metric_values = [class_metrics["precision"], class_metrics["recall"], class_metrics["f1_score"], class_metrics["accuracy"]]
        colors = ["#3498db", "#e74c3c", "#f39c12", "#2ecc71"]

        bars = ax1.bar(metric_names, metric_values, color=colors, alpha=0.8)
        ax1.set_ylim(0, 1)
        ax1.set_ylabel("Score")
        ax1.grid(axis="y", alpha=0.3)
        for bar, value in zip(bars, metric_values):
            ax1.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.3f}", ha="center", va="bottom", fontweight="bold")

        breakdown = [class_metrics["true_positives"], class_metrics["true_negatives"], class_metrics["false_positives"], class_metrics["false_negatives"]]
        labels = ["TP", "TN", "FP", "FN"]
        bars2 = ax2.bar(labels, breakdown, color=["#27ae60", "#2980b9", "#f39c12", "#c0392b"], alpha=0.8)
        ax2.set_ylabel("Count")
        ax2.grid(axis="y", alpha=0.3)
        for bar, value in zip(bars2, breakdown):
            ax2.text(bar.get_x() + bar.get_width() / 2, value + max(1, value * 0.02), str(int(value)), ha="center", va="bottom", fontweight="bold")

        plt.tight_layout(rect=[0, 0, 1, 0.94])
        fig.savefig(os.path.join(plots_dir, f"{dataset_name}_classification_metrics.png"), dpi=300)
        plt.close(fig)

    def _plot_protection_analysis(self, metrics: Dict[str, Any], plots_dir: str, dataset_name: str) -> None:
        protection = metrics["protection_metrics"]
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f"{dataset_name} • Network Protection Analysis", fontweight="bold")

        axes[0, 0].bar(["Trust-Based", "Baseline"], [protection["trust_total_attacks"], protection["baseline_total_attacks"]], color=["#27ae60", "#e74c3c"], alpha=0.8)
        axes[0, 0].set_ylabel("Attack Attempts")
        axes[0, 0].grid(axis="y", alpha=0.3)

        axes[0, 1].bar(["Trust-Based", "Baseline"], [protection["trust_successful_attacks"], protection["baseline_successful_attacks"]], color=["#27ae60", "#e74c3c"], alpha=0.8)
        axes[0, 1].set_ylabel("Successful Attacks")
        axes[0, 1].grid(axis="y", alpha=0.3)

        prevented = max(0, protection["attacks_prevented"])
        pie_labels = ["Prevented", "Still Successful"] if protection["baseline_successful_attacks"] else ["Prevented"]
        pie_sizes = [prevented, max(0, protection["trust_successful_attacks"])] if protection["baseline_successful_attacks"] else [prevented]
        if sum(pie_sizes) > 0:
            axes[1, 0].pie(pie_sizes, labels=pie_labels, autopct="%1.1f%%", startangle=90, colors=["#2ecc71", "#e74c3c"])
            axes[1, 0].set_title("Attack Prevention Share")
        else:
            axes[1, 0].text(0.5, 0.5, "No attacks recorded", ha="center", va="center", fontsize=12)
            axes[1, 0].set_axis_off()

        axes[1, 1].bar(["Prevention Rate"], [protection["prevention_rate"] * 100], color="#9b59b6", alpha=0.8)
        axes[1, 1].set_ylim(0, 100)
        axes[1, 1].set_ylabel("% Prevented")
        axes[1, 1].grid(axis="y", alpha=0.3)

        plt.tight_layout(rect=[0, 0, 1, 0.94])
        fig.savefig(os.path.join(plots_dir, f"{dataset_name}_protection_analysis.png"), dpi=300)
        plt.close(fig)

    def _plot_performance_comparison(self, metrics: Dict[str, Any], plots_dir: str, dataset_name: str) -> None:
        phases = metrics["phases"]
        labels = [phases[p]["phase_name"] for p in ["training", "testing", "trust_based", "baseline"]]
        success_rates = [phases[p]["success_rate"] for p in ["training", "testing", "trust_based", "baseline"]]
        latencies = [phases[p]["avg_latency"] for p in ["training", "testing", "trust_based", "baseline"]]
        energies = [phases[p]["energy_consumption"] for p in ["training", "testing", "trust_based", "baseline"]]

        improvements = metrics["improvements"]
        improvement_names = ["Success Rate Δ", "Latency Δ", "Energy Δ"]
        improvement_values = [improvements["success_rate_improvement"], improvements["latency_improvement"], improvements["energy_improvement"]]

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f"{dataset_name} • Performance Comparison", fontweight="bold")

        axes[0, 0].bar(labels, success_rates, color="#1abc9c", alpha=0.8)
        axes[0, 0].set_ylabel("Success Rate")
        axes[0, 0].set_ylim(0, 1)
        axes[0, 0].grid(axis="y", alpha=0.3)

        axes[0, 1].bar(labels, latencies, color="#e67e22", alpha=0.8)
        axes[0, 1].set_ylabel("Average Latency")
        axes[0, 1].grid(axis="y", alpha=0.3)

        axes[1, 0].bar(labels, energies, color="#9b59b6", alpha=0.8)
        axes[1, 0].set_ylabel("Total Energy")
        axes[1, 0].grid(axis="y", alpha=0.3)

        axes[1, 1].bar(improvement_names, improvement_values, color="#34495e", alpha=0.8)
        axes[1, 1].set_ylabel("Improvement")
        axes[1, 1].grid(axis="y", alpha=0.3)

        plt.tight_layout(rect=[0, 0, 1, 0.94])
        fig.savefig(os.path.join(plots_dir, f"{dataset_name}_performance_comparison.png"), dpi=300)
        plt.close(fig)

    def _plot_trust_distribution(self, metrics: Dict[str, Any], plots_dir: str, dataset_name: str) -> None:
        analysis = metrics["trust_analysis"]
        network_info = metrics["network_info"]

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(f"{dataset_name} • Trust Distribution", fontweight="bold")

        malicious_array = np.full(network_info["malicious_nodes"], analysis["malicious_trust_mean"]) if network_info["malicious_nodes"] else np.array([])
        honest_array = np.full(network_info["honest_nodes"], analysis["honest_trust_mean"]) if network_info["honest_nodes"] else np.array([])

        data = []
        labels = []
        if malicious_array.size:
            data.append(malicious_array)
            labels.append("Malicious")
        if honest_array.size:
            data.append(honest_array)
            labels.append("Honest")

        if data:
            axes[0].boxplot(data, labels=labels, patch_artist=True)
            axes[0].set_ylabel("Trust Value")
            axes[0].grid(axis="y", alpha=0.3)
        else:
            axes[0].text(0.5, 0.5, "No trust data available", ha="center", va="center", fontsize=12)
            axes[0].set_axis_off()

        total_nodes = network_info["honest_nodes"] + network_info["malicious_nodes"]
        if total_nodes > 0:
            axes[1].pie([network_info["honest_nodes"], network_info["malicious_nodes"]], labels=["Honest", "Malicious"], autopct="%1.1f%%", startangle=90, colors=["#2ecc71", "#e74c3c"])
            axes[1].set_title(f"Trust Gap: {analysis['trust_gap']:.3f} ({analysis['separation_quality']})")
        else:
            axes[1].text(0.5, 0.5, "No nodes available", ha="center", va="center", fontsize=12)
            axes[1].set_axis_off()

        plt.tight_layout(rect=[0, 0, 1, 0.94])
        fig.savefig(os.path.join(plots_dir, f"{dataset_name}_trust_distribution.png"), dpi=300)
        plt.close(fig)

    def _plot_confusion_matrix(self, metrics: Dict[str, Any], plots_dir: str, dataset_name: str) -> None:
        cm = np.array(metrics["classification_metrics"]["confusion_matrix"])
        if cm.shape != (2, 2):
            return

        fig, ax = plt.subplots(figsize=(6, 6))
        fig.suptitle(f"{dataset_name} • Confusion Matrix", fontweight="bold")
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Honest", "Malicious"], yticklabels=["Honest", "Malicious"], ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(os.path.join(plots_dir, f"{dataset_name}_confusion_matrix.png"), dpi=300)
        plt.close(fig)

    # ------------------------------------------------------------------
    # Summaries & reporting
    # ------------------------------------------------------------------
    def _save_metrics(self, dataset_dir: str, metrics: Dict[str, Any]) -> None:
        metrics_path = os.path.join(dataset_dir, "quick_comprehensive_metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as fh:
            json.dump(self._convert_for_json(metrics), fh, indent=2)

    def _generate_summary(self, all_results: Dict[str, Dict[str, Any]]) -> None:
        summary = {
            "timestamp": datetime.now().isoformat(),
            "total_datasets": len(all_results),
            "successful_evaluations": len([1 for r in all_results.values() if "error" not in r]),
            "failed_evaluations": len([1 for r in all_results.values() if "error" in r])
        }

        valid = [r for r in all_results.values() if "error" not in r]
        if valid:
            summary.update({
                "avg_testing_success": float(np.mean([r["testing_success_rate"] for r in valid])),
                "avg_trust_success": float(np.mean([r["trust_success_rate"] for r in valid])),
                "avg_trust_improvement": float(np.mean([r["trust_improvement"] for r in valid])),
                "avg_precision": float(np.mean([r["precision"] for r in valid])),
                "avg_recall": float(np.mean([r["recall"] for r in valid]))
            })

        summary_path = os.path.join(self.results_dir, "quick_evaluation_summary.json")
        with open(summary_path, "w", encoding="utf-8") as fh:
            json.dump(self._convert_for_json(summary), fh, indent=2)

        print("\n📊 Quick Evaluation Summary")
        print(f"   ✅ Success: {summary['successful_evaluations']}/{summary['total_datasets']}")
        if valid:
            print(f"   📈 Avg testing success: {summary['avg_testing_success']:.2%}")
            print(f"   🛡️ Avg trust-based success: {summary['avg_trust_success']:.2%}")
            print(f"   ✨ Avg trust improvement: {summary['avg_trust_improvement']:.3f}")
        print(f"📄 Summary JSON: {summary_path}")

    def _generate_html_report(self) -> None:
        if not self.dataset_metrics:
            return

        html_path = os.path.join(self.results_dir, "quick_all_datasets_report.html")
        html_sections = []

        for dataset_name, metrics in self.dataset_metrics.items():
            class_metrics = metrics["classification_metrics"]
            protection = metrics["protection_metrics"]
            improvements = metrics["improvements"]

            html_sections.append(f"""
        <div class="dataset">
            <h2>📊 {dataset_name.upper().replace('_', ' • ')}</h2>
            <h3>🎯 Detection Metrics</h3>
            <div class="metric-grid">
                <div class="metric-card"><h4>Precision</h4><span>{class_metrics['precision']:.3f}</span></div>
                <div class="metric-card"><h4>Recall</h4><span>{class_metrics['recall']:.3f}</span></div>
                <div class="metric-card"><h4>F1-Score</h4><span>{class_metrics['f1_score']:.3f}</span></div>
                <div class="metric-card"><h4>Accuracy</h4><span>{class_metrics['accuracy']:.3f}</span></div>
            </div>

            <h3>🛡️ Network Protection</h3>
            <table>
                <tr><th>Metric</th><th>Trust-Based</th><th>Baseline</th><th>Improvement</th></tr>
                <tr><td>Total Attacks</td><td>{protection['trust_total_attacks']}</td><td>{protection['baseline_total_attacks']}</td><td>-</td></tr>
                <tr><td>Successful Attacks</td><td>{protection['trust_successful_attacks']}</td><td>{protection['baseline_successful_attacks']}</td><td>{protection['attacks_prevented']} prevented</td></tr>
                <tr><td>Prevention Rate</td><td colspan="3">{protection['prevention_rate']*100:.1f}%</td></tr>
            </table>

            <h3>🚀 Performance Improvements</h3>
            <div class="metric-grid">
                <div class="metric-card"><h4>Success Rate Δ</h4><span>{improvements['success_rate_improvement']:.3f}</span></div>
                <div class="metric-card"><h4>Latency Δ</h4><span>{improvements['latency_improvement']:.2f}</span></div>
                <div class="metric-card"><h4>Energy Δ</h4><span>{improvements['energy_improvement']:.2f}</span></div>
                <div class="metric-card"><h4>Relative Gain</h4><span>{improvements['improvement_percentage']:.2f}%</span></div>
            </div>

            <h3>📈 Visualizations</h3>
            <div class="viz-grid">
                {self._html_plot_block(dataset_name, 'trust_trajectories', 'Trust Trajectories During Attacks')}
                {self._html_plot_block(dataset_name, 'loss_curves', 'Training & Validation Loss Curves')}
                {self._html_plot_block(dataset_name, 'attack_timeline', 'Attack Timeline & Detection')}
                {self._html_plot_block(dataset_name, 'classification_metrics', 'Precision / Recall / F1')}
                {self._html_plot_block(dataset_name, 'protection_analysis', 'Network Protection Analysis')}
                {self._html_plot_block(dataset_name, 'confusion_matrix', 'Confusion Matrix')}
                {self._html_plot_block(dataset_name, 'performance_comparison', 'Performance Comparison')}
                {self._html_plot_block(dataset_name, 'trust_distribution', 'Trust Distribution & Composition')}
            </div>
        </div>
            """)

        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Quick Comprehensive Mid-Semester Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; background: #f7f8fb; margin: 0; padding: 0 0 40px 0; }}
        .header {{ background: linear-gradient(135deg, #1f8ef1 0%, #5f27cd 100%); color: white; padding: 40px; text-align: center; }}
        .container {{ width: 95%; margin: 0 auto; }}
        .dataset {{ background: #fff; margin: 25px 0; padding: 25px; border-radius: 12px; box-shadow: 0 6px 18px rgba(0,0,0,0.07); }}
        h1 {{ margin-bottom: 10px; }}
        h2 {{ border-bottom: 2px solid #1f8ef1; padding-bottom: 10px; color: #222; }}
        h3 {{ color: #333; margin-top: 25px; }}
        .metric-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 15px; }}
        .metric-card {{ background: #f4f7ff; border-radius: 8px; padding: 18px; text-align: center; color: #333; box-shadow: inset 0 0 0 1px rgba(31,142,241,0.05); }}
        .metric-card h4 {{ margin-bottom: 10px; font-size: 16px; color: #444; }}
        .metric-card span {{ font-size: 24px; font-weight: bold; color: #1f8ef1; }}
        .viz-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 18px; margin-top: 15px; }}
        .viz-item {{ background: #fafbff; border-radius: 10px; padding: 12px; text-align: center; box-shadow: inset 0 0 0 1px rgba(31,142,241,0.05); }}
        .viz-item img {{ max-width: 100%; border-radius: 6px; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 12px; }}
        th, td {{ border: 1px solid #e0e6f3; padding: 10px; text-align: center; }}
        th {{ background: #1f8ef1; color: white; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Quick Comprehensive Mid-Semester GNN Trust Report</h1>
        <p>Trust trajectories • Loss curves • Attack timelines • Precision/Recall/F1 • Network protection</p>
        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    <div class="container">
        {''.join(html_sections)}
    </div>
</body>
</html>
        """

        with open(html_path, "w", encoding="utf-8") as fh:
            fh.write(html_content)

        print(f"🌐 HTML dashboard: {html_path}")

    def _html_plot_block(self, dataset_name: str, suffix: str, title: str) -> str:
        relative_path = f"{dataset_name}/plots/{dataset_name}_{suffix}.png"
        return f"""
        <div class="viz-item">
            <img src="{relative_path}" alt="{title}">
            <p><strong>{title}</strong></p>
        </div>
        """

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def _pad_curve(self, values: List[float], target_length: int, fill_value: float = np.nan) -> List[float]:
        values = list(values)
        if len(values) >= target_length:
            return values[:target_length]
        return values + [fill_value] * (target_length - len(values))

    def _convert_for_json(self, data: Any) -> Any:
        if isinstance(data, dict):
            return {str(k): self._convert_for_json(v) for k, v in data.items()}
        if isinstance(data, list):
            return [self._convert_for_json(v) for v in data]
        if isinstance(data, (np.integer, np.int8, np.int16, np.int32, np.int64)):
            return int(data)
        if isinstance(data, (np.floating, np.float16, np.float32, np.float64)):
            return float(data)
        if isinstance(data, (np.ndarray, pd.Series)):
            return self._convert_for_json(data.tolist())
        if isinstance(data, (datetime,)):
            return data.isoformat()
        if isinstance(data, float):
            if math.isinf(data) or math.isnan(data):
                return None
        if isinstance(data, (np.bool_, bool)):
            return bool(data)
        return data


def trajectory_window(timeline: List[int]) -> int:
    if len(timeline) < 2:
        return 1
    diffs = np.diff(sorted(timeline))
    diff = int(np.median(diffs)) if diffs.size else 1
    return max(diff, 1)


def main() -> None:
    evaluator = QuickAllDatasetsEvaluator()
    evaluator.run_all_datasets()


if __name__ == "__main__":
    main()