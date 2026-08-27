from loguru import logger


def register():
    """注册 vllm_pd 策略（P/D 分离集群仿真）。"""
    from optix.optimizer.register import register_simulator
    from vllm_pd_simulator.pd_cluster_simulator import PdClusterSimulator

    register_simulator("vllm_pd", PdClusterSimulator)
    logger.info("vllm_pd_simulator: registered (vllm_pd)")
