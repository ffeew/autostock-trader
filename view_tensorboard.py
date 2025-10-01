"""
Helper script to launch TensorBoard.

Makes it easy to view training visualizations.
"""

import os
import sys
import argparse
import subprocess

def main():
    parser = argparse.ArgumentParser(description='Launch TensorBoard to view training logs')
    parser.add_argument(
        '--logdir',
        type=str,
        default='runs',
        help='Directory containing TensorBoard logs (default: runs)'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=6006,
        help='Port to run TensorBoard on (default: 6006)'
    )
    parser.add_argument(
        '--host',
        type=str,
        default='localhost',
        help='Host to bind TensorBoard to (default: localhost)'
    )

    args = parser.parse_args()

    # Check if log directory exists
    if not os.path.exists(args.logdir):
        print(f"❌ Error: Log directory '{args.logdir}' does not exist.")
        print("\nMake sure you've trained models with --tensorboard flag:")
        print("  python train_models.py --tensorboard")
        sys.exit(1)

    # Check if directory has any logs
    log_count = 0
    for root, dirs, files in os.walk(args.logdir):
        for file in files:
            if file.startswith('events.out.tfevents'):
                log_count += 1
                break
        if log_count > 0:
            break

    if log_count == 0:
        print(f"⚠️  Warning: No TensorBoard logs found in '{args.logdir}'")
        print("\nRun training with --tensorboard to generate logs:")
        print("  python train_models.py --tensorboard")
        sys.exit(1)

    # Launch TensorBoard
    print("="*60)
    print("Launching TensorBoard")
    print("="*60)
    print(f"\nLog directory: {args.logdir}")
    print(f"URL: http://{args.host}:{args.port}")
    print("\nPress Ctrl+C to stop TensorBoard\n")

    try:
        subprocess.run([
            'tensorboard',
            '--logdir', args.logdir,
            '--port', str(args.port),
            '--host', args.host
        ])
    except KeyboardInterrupt:
        print("\n\nTensorBoard stopped.")
    except FileNotFoundError:
        print("❌ Error: TensorBoard not found. Install it with:")
        print("  pip install tensorboard")
        sys.exit(1)


if __name__ == '__main__':
    main()
