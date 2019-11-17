import argparse
from datetime import datetime
import os

from sagemaker.estimator import Estimator

MODEL_NAME = 'ssd'
BUCKET_NAME = 'your_bucket_name'
HYPER_PARAMS = {
    'output_path': '/opt/ml/model',
    'ckpt_path': '/opt/ml/checkpoints',
    'data_dir': '/opt/ml/input/data/training'
}
IAM_ROLE = os.getenv('AWS_IAM_ROLE')
MAX_RUN_TIME = 24 * 60 * 60 * 3  # 3 days


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', '-d', type=str, required=True,
                        help='s3 path of training dataset (e.g. s3://...')
    parser.add_argument('--image_name', '-m', type=str, required=True)
    parser.add_argument('--instance_type', '-i', type=str, default='ml.p2.xlarge')

    return parser.parse_args()


def main():
    args = get_args()

    job_name = MODEL_NAME + '-' + datetime.now().strftime('%Y-%m-%d-%H%M%S')
    output_path = 's3://{}/{}/jobs/'.format(BUCKET_NAME, MODEL_NAME)

    # local mode does not support spot training.
    is_spot = args.instance_type != 'local'
    checkpoint_s3_path = '{}{}/checkpoint/'.format(output_path, job_name) if is_spot else None

    estimator = Estimator(
        image_name=args.image_name,
        role=IAM_ROLE,
        train_instance_count=1,
        train_instance_type=args.instance_type,
        train_max_run=MAX_RUN_TIME,
        train_max_wait=MAX_RUN_TIME,
        output_path=output_path,
        train_use_spot_instances=is_spot,
        checkpoint_s3_uri=checkpoint_s3_path,
        hyperparameters=HYPER_PARAMS)

    estimator.fit(inputs=args.dataset_path, job_name=job_name)


if __name__ == '__main__':
    main()
