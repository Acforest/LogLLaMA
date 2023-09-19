from huggingface_hub import snapshot_download


repo_id = "meta-llama/Llama-2-70b-chat-hf"  # 模型在huggingface上的名称
local_dir = "./Llama-2-70b-chat-hf/"  # 本地模型存储的地址
local_dir_use_symlinks = False  # 本地模型使用文件保存，而非blob形式保存
token = "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"  # 在hugging face上生成的 access token

# 如果需要代理的话
# proxies = {
#     'http': 'XXXX',
#     'https': 'XXXX',
# }

snapshot_download(
    repo_id=repo_id,
    local_dir=local_dir,
    local_dir_use_symlinks=local_dir_use_symlinks,
    token=token,
    # proxies=proxies
)
