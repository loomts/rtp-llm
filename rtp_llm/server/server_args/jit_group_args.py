def init_jit_group_args(parser, jit_config):
    ##############################################################################################################
    # JIT Configuration
    ##############################################################################################################
    jit_group = parser.add_argument_group("JIT Configuration")
    jit_group.add_argument(
        "--remote_jit_dir",
        env_name="REMOTE_JIT_DIR",
        bind_to=(jit_config, "remote_jit_dir"),
        type=str,
        default="",
        help="JIT远程cache目录",
    )
    jit_group.add_argument(
        "--local_jit_cache_dir",
        env_name="LOCAL_JIT_CACHE_DIR",
        bind_to=(jit_config, "local_jit_cache_dir"),
        type=str,
        default="~/.cache/rtp_llm_jit",
        help="JIT本地cache目录",
    )
    jit_group.add_argument(
        "--jit_prepare_timeout_s",
        env_name="JIT_PREPARE_TIMEOUT_S",
        bind_to=(jit_config, "jit_prepare_timeout_s"),
        type=int,
        default=30,
        help="启动路径等待JIT remote cache预热的最长时间",
    )
    jit_group.add_argument(
        "--jit_sync_interval_s",
        env_name="JIT_SYNC_INTERVAL_S",
        bind_to=(jit_config, "jit_sync_interval_s"),
        type=int,
        default=300,
        help="JIT cache后台写回间隔",
    )
