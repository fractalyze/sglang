W=/opt/dlami/nvme/work/kernels-zgsyac
git -C $W fetch -q origin draft/kernels-zgsyac && git -C $W checkout -q -f FETCH_HEAD
git -C $W log --oneline -1
R=results/$(date -u +%Y%m%dT%H%M%SZ)-t5; mkdir -p $W/$R
echo RESULT_DIR=$W/$R
IMG=810421502523.dkr.ecr.us-east-2.amazonaws.com/dsv32/sglang:current
gpu-lease 1 --wait 1800 -- bash -c "docker run --rm --gpus device=\$CUDA_VISIBLE_DEVICES -v $W:/w -v $W/cache:/root/.cache -e PYTHONPATH=/w/python -e R=$R -w /w $IMG bash -lc 'timeout 3000 python3 -m pytest -q -p no:cacheprovider --tb=line -rf test/registered/kernels/ops/moe/test_w4a16_moe_sm90.py > $R/pytest.txt 2>&1; timeout 3000 python3 test/manual/kernels/bench_w4a16_moe_sm90.py --sweep-blocks --tokens-per-expert 4 8 16 32 64 --out $R/bench.jsonl > $R/bench.log 2>&1'"
for f in $W/$R/*; do echo "=== $f"; tail -c 9000 $f; done
aws s3 sync $W/$R s3://fractalyze-dsv32-use2/runs/kernels-zgsyac-kernel/$(basename $R)/ --only-show-errors
