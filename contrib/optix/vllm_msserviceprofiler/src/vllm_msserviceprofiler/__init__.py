def register():
    from vllm_msserviceprofiler.simulator import CustomVllmDockerSimulator
    from vllm_msserviceprofiler.settings import CusSettings
    from optix.optimizer.register import register_simulator
    from optix.config.config import register_settings

    register_simulator("custom_vllm", CustomVllmDockerSimulator)
    register_settings(lambda: CusSettings())
