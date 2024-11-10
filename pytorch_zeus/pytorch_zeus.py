import torch
from transformers import DistilBertModel, DistilBertTokenizer
from datasets import load_dataset
from zeus.monitor import ZeusMonitor
from zeus.device.cpu import get_current_cpu_index
import pandas as pd
from typing import Dict, List
from idistilbert import iDistilBertModel
import platform
import os

class EnergyBenchmark:
    def __init__(self, batch_sizes: List[int], samples_per_measurement: int = 1000, force_cpu: bool = False):
        self.batch_sizes = batch_sizes
        self.samples_per_measurement = samples_per_measurement
        self.force_cpu = force_cpu
        
        # Initialize Zeus monitor with minimal configuration first
        try:
            if torch.cuda.is_available() and not force_cpu:
                self.monitor = ZeusMonitor(
                    gpu_indices=[torch.cuda.current_device()],
                    cpu_indices=[]  # Don't monitor CPU for now
                )
                print("Zeus monitor initialized with GPU only")
                self.device = torch.device('cuda')
            else:
                self.monitor = ZeusMonitor(
                    cpu_indices=[],  # Don't monitor CPU for now
                    gpu_indices=[]
                )
                print("Zeus monitor initialized without CPU/GPU")
                self.device = torch.device('cpu')
        except Exception as e:
            print(f"Warning: Zeus monitor initialization failed: {e}")
            print("Running without energy measurements")
            self.monitor = None
            self.device = torch.device('cpu') if force_cpu else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Using device: {self.device}")
        print(f"CPU Type: {platform.processor()}")
        if torch.cuda.is_available() and not force_cpu:
            print(f"GPU Type: {torch.cuda.get_device_name()}")
        
        # Load tokenizer
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        
        # Load datasets
        print("Loading STSB dataset...")
        dataset = load_dataset('glue', 'stsb')
        self.all_sentences = (
            dataset['train']['sentence1'] + 
            dataset['validation']['sentence1']
        )
        print(f"Total available samples: {len(self.all_sentences)}")
        
        # Initialize models
        print("Loading models...")
        self.dot_product_model = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.inhibitor_model = iDistilBertModel.from_pretrained('tonytonfisk/inhibitor_distilbert')
        
        self.dot_product_model.to(self.device)
        self.inhibitor_model.to(self.device)
        
        # Set models to eval mode
        self.dot_product_model.eval()
        self.inhibitor_model.eval()
    
    def prepare_batches(self, batch_size: int, num_samples: int) -> List[Dict[str, torch.Tensor]]:
        batches = []
        for start_idx in range(0, num_samples, batch_size):
            actual_indices = [(start_idx + i) % len(self.all_sentences) 
                            for i in range(batch_size)]
            sentences = [self.all_sentences[i] for i in actual_indices]
            
            inputs = self.tokenizer(
                sentences,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors='pt'
            )
            batches.append({k: v.to(self.device) for k, v in inputs.items()})
        return batches
    
    def measure_continuous_energy(self, model, batches: List[Dict[str, torch.Tensor]], 
                                model_name: str, batch_size: int) -> Dict:
        if self.monitor is None:
            for batch in batches:
                with torch.no_grad():
                    model(**batch)
            return {'time': 0.0, 'total_energy': 0.0}
        
        total_samples = len(batches) * batch_size
        print(f"  Processing {total_samples} samples in {len(batches)} batches...")
        
        # Start energy measurement
        window_name = f"{model_name}_batch{batch_size}"
        try:
            self.monitor.begin_window(window_name)
        except Exception as e:
            print(f"Warning: Energy measurement failed to start: {e}")
            print("Continuing without energy measurement...")
            for batch in batches:
                with torch.no_grad():
                    model(**batch)
            return {'time': 0.0, 'total_energy': 0.0}
        
        # Process all batches continuously
        try:
            with torch.no_grad():
                for i, batch in enumerate(batches):
                    model(**batch)
                    if (i + 1) % 10 == 0:
                        print(f"    Processed {(i + 1) * batch_size} samples")
            
            # End energy measurement
            measurement = self.monitor.end_window(window_name)
            return measurement
        except Exception as e:
            print(f"Warning: Energy measurement failed: {e}")
            print("Returning zero measurements...")
            return {'time': 0.0, 'total_energy': 0.0}
    
    def run_benchmark(self, num_runs: int = 3) -> pd.DataFrame:
        all_results = []
        
        # Warmup
        print("\nPerforming warmup...")
        warmup_batches = self.prepare_batches(32, 128)  # Small warmup
        for batch in warmup_batches:
            with torch.no_grad():
                self.dot_product_model(**batch)
                self.inhibitor_model(**batch)
        
        # Main measurements
        for batch_size in self.batch_sizes:
            print(f"\nRunning benchmark for batch_size={batch_size}")
            
            for run in range(num_runs):
                print(f"\nRun {run + 1}/{num_runs}")
                
                # Prepare batches for this run
                batches = self.prepare_batches(batch_size, self.samples_per_measurement)
                
                print("  Measuring dot product attention...")
                dot_prod_mes = self.measure_continuous_energy(
                    self.dot_product_model, batches, "dot_prod", batch_size
                )
                
                print("  Measuring inhibitor attention...")
                inhib_mes = self.measure_continuous_energy(
                    self.inhibitor_model, batches, "inhib", batch_size
                )
                
                # Record results
                for mes, model_type in [(dot_prod_mes, 'dot_product'), 
                                      (inhib_mes, 'inhibitor')]:
                    if isinstance(mes, dict):
                        result = {
                            'batch_size': batch_size,
                            'run': run,
                            'model_type': model_type,
                            'time': mes.get('time', 0.0),
                            'energy': mes.get('total_energy', 0.0),
                            'samples_processed': self.samples_per_measurement,
                            'energy_per_sample': mes.get('total_energy', 0.0) / self.samples_per_measurement,
                            'throughput': self.samples_per_measurement / mes.get('time', 1.0) if mes.get('time', 0.0) > 0 else 0.0
                        }
                    else:
                        result = {
                            'batch_size': batch_size,
                            'run': run,
                            'model_type': model_type,
                            'time': getattr(mes, 'time', 0.0),
                            'energy': getattr(mes, 'total_energy', 0.0),
                            'samples_processed': self.samples_per_measurement,
                            'energy_per_sample': getattr(mes, 'total_energy', 0.0) / self.samples_per_measurement,
                            'throughput': self.samples_per_measurement / getattr(mes, 'time', 1.0) if getattr(mes, 'time', 0.0) > 0 else 0.0
                        }
                    
                    all_results.append(result)
                
                print(f"  Completed run {run + 1} for batch size {batch_size}")
                print(f"  Total samples processed: {self.samples_per_measurement}")
        
        return pd.DataFrame(all_results)

def analyze_results(df: pd.DataFrame) -> None:
    print("\nAnalyzing results...")
    
    # Get all energy-related columns
    energy_cols = [col for col in df.columns if 'energy' in col.lower()]
    
    # Group by model type and batch size
    grouped = df.groupby(['model_type', 'batch_size'])
    
    # Calculate statistics
    metrics = ['time', 'throughput'] + energy_cols
    stats = grouped[metrics].agg(['mean', 'std']).round(4)
    
    print("\nDetailed Statistics:")
    print(stats)
    
    # Calculate relative efficiency for each batch size
    for batch_size in df['batch_size'].unique():
        dot_prod = df[(df['model_type'] == 'dot_product') & 
                     (df['batch_size'] == batch_size)]
        
        inhib = df[(df['model_type'] == 'inhibitor') & 
                   (df['batch_size'] == batch_size)]
        
        print(f"\nBatch Size: {batch_size}")
        print(f"Samples per measurement: {dot_prod['samples_processed'].iloc[0]}")
        
        print(f"Time (seconds):")
        print(f"  Dot Product: {dot_prod['time'].mean():.4f} ± {dot_prod['time'].std():.4f}")
        print(f"  Inhibitor: {inhib['time'].mean():.4f} ± {inhib['time'].std():.4f}")
        
        print(f"Throughput (samples/second):")
        print(f"  Dot Product: {dot_prod['throughput'].mean():.2f} ± {dot_prod['throughput'].std():.2f}")
        print(f"  Inhibitor: {inhib['throughput'].mean():.2f} ± {inhib['throughput'].std():.2f}")
        
        print(f"Energy (Joules):")
        print(f"  Dot Product: {dot_prod['energy'].mean():.4f} ± {dot_prod['energy'].std():.4f}")
        print(f"  Inhibitor: {inhib['energy'].mean():.4f} ± {inhib['energy'].std():.4f}")
        
        print(f"Energy per Sample (Joules/sample):")
        print(f"  Dot Product: {dot_prod['energy_per_sample'].mean():.4f} ± {dot_prod['energy_per_sample'].std():.4f}")
        print(f"  Inhibitor: {inhib['energy_per_sample'].mean():.4f} ± {inhib['energy_per_sample'].std():.4f}")
        
        if dot_prod['energy'].mean() != 0:
            relative_efficiency = (dot_prod['energy'].mean() - inhib['energy'].mean()) / dot_prod['energy'].mean() * 100
            print(f"Relative Energy Efficiency: {relative_efficiency:.2f}% " + 
                  f"({'more' if relative_efficiency > 0 else 'less'} efficient)")
    
    return stats

if __name__ == "__main__":
    try:
        batch_sizes = [1, 8, 16, 32]
        
        # Only run GPU benchmark
        if torch.cuda.is_available():
            print("\n=== Running GPU Benchmark ===")
            benchmark_gpu = EnergyBenchmark(
                batch_sizes=batch_sizes,
                samples_per_measurement=1000,
                force_cpu=False
            )
            
            print("\nStarting GPU benchmark runs...")
            results_gpu = benchmark_gpu.run_benchmark(num_runs=3)
            stats_gpu = analyze_results(results_gpu)
            
            print("\nSaving GPU results...")
            results_gpu.to_csv('energy_results_gpu.csv')
            stats_gpu.to_csv('energy_stats_gpu.csv')
        
        print("\nBenchmark complete!")
        
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        raise