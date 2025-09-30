import os
import sys
import torch
import datetime
import argparse
from ml4co.Trainer.MIPtrainer import MIPTrain
from ml4co.DataCollector.MIPdata import MIPData

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
def main():

    parser = argparse.ArgumentParser(description="A script to handle size and instances parameters")

    parser.add_argument('task', type=str, nargs='?', default='node_selection', help='task type (default: branch)')
    parser.add_argument('problem', type=str, nargs='?', default='cauctions', help='Instance type (default: setcover)')
    parser.add_argument('method', type=str, nargs='?', default='gnn', help='Testing methods (default: lns_CL)')
    
    args = parser.parse_args()

    DATA_DIR =  os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    MODEL_DIR =  os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
    DEVICE = torch.device('cuda:6' if torch.cuda.is_available() else 'cpu')
    
    print(DATA_DIR, MODEL_DIR, DEVICE)

    MIPTrainer = MIPTrain(model_dir=MODEL_DIR, data_dir=DATA_DIR, device=DEVICE)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print(f"[{timestamp}] Starting training | Method: {args.method} | Problem: {args.problem}")
    
    model_path = MIPTrainer.train(task=args.task, problem=args.problem, method=args.method)
    print(model_path)

    print(f"[{timestamp}] Finished training | Task: {args.task} |Method: {args.method} | Problem: {args.problem}")

if __name__ == '__main__':

    main()