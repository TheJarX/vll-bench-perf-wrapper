# Yet another wrapper for a tool that really doesn't need it

This bash scripts aims to make easier to benchmark performance of LLMs using vLLM `bench`.

## Executing
**Note:** You can always just make the script executable instead. (`chmod +x bench.sh`)
```bash
bash bench.sh --models /local/path/to/model --vllm-extra-args "-tp 4 --dtype half"
```

You can also generate the output directly using the python script:
```bash
python process_results.py /path/to/results
```
> You can get a list of the available flags using `bash bench.sh --help`