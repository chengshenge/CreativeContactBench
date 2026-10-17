# 给新电脑 Codex 的任务

请在这个独立运行包中实际完成四个Qwen的RGB30 BF16实验。先读README.md、run_plan.json、configs/*.json、VALIDATION.md。

1. 这是30题主榜扩展，不是67题默认evaluation；不读旧答案来补全新结果，不修改论文、旧实验、任务或人评。只使用包内已冻结的30题图片和prompt，v1.2、canonical A–D、每题一次。
2. 核对nvidia-smi、nvidia-smi topo -m、可用RAM/磁盘、Python/驱动。使用作者预留的两张48GB卡；其他卡及其他人的进程保持不动。
3. 按README建立独立runner/server环境，执行verify及全部离线测试。按四个配置固定的官方revision下载BF16权重。不能替换版本、量化、把Instruct换成Thinking或热修复冻结输入。依赖兼容问题明确报告BLOCKED。
4. 依次启动serve，等服务就绪，执行synthetic smoke，再执行run。前三个thinking，8B保持Instruct。正式输出预算已固定。120个正式槽位；有上限的瞬态重试由脚本控制。回答不好、拒答、字段缺失、输出截断、格式错误均不能额外重跑。
5. 部署成功不等于官方已验证本硬件；保留配置、文件hash、server.log、环境与GPU状态。模型权重或架构不支持时报告，不擅自换模型。禁止读取任何密钥或输出token。无需付费API。
6. 完成后执行analyze和pack-results，给作者返回rgb30_bf16_results.zip及完成/无效/缺失数量。结果不好也照实保留。主表合并交回原机器完成，不覆盖原量化行。

运行包已经通过离线历史回归验证，但没有在这台目标GPU机器执行过BF16推理；现在需要你完成部署验证与正式运行。所有相关命令见README。
