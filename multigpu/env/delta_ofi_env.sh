# NCCL over Slingshot (HPE cxi) for JAX's pip-bundled NCCL on NCSA Delta GH200. Sourced by setup_delta.sh.
# Adds the aws-ofi-nccl plugin + Cray libfabric to LD_LIBRARY_PATH (found by NCCL as libnccl-net.so), plus Delta's
# FI_CXI_* / NCCL_* settings. Does NOT put HPC-SDK CUDA/NCCL on the path (that causes "free(): double free").
#   OFI_PLUGIN=1.18 (default on >1 node) | 1.14 | none (TCP sockets; default on 1 node, where NVLink is used anyway)
# Needs $PY (the env's python) to locate the pip CUDA runtime/cupti libs the 1.18 plugin links against.
_SP=$("$PY" -c 'import nvidia, os; print(list(nvidia.__path__)[0])' 2>/dev/null)
if [ -z "${OFI_PLUGIN:-}" ]; then
  if [ "${SLURM_JOB_NUM_NODES:-1}" -gt 1 ]; then OFI_PLUGIN=1.18; else OFI_PLUGIN=none; fi
fi
case "$OFI_PLUGIN" in
  1.18) _PLUG=/sw/user/nccl/aws-ofi-nccl-1.18.0-lf2.3.1-cu12.9/lib ;;
  1.14) _PLUG=/sw/user/nccl/aws-ofi-nccl-1.14.2-lf2.3.1-cu12.9/lib ;;
  none) _PLUG= ;;
  *) echo "unknown OFI_PLUGIN=$OFI_PLUGIN" >&2; return 1 ;;
esac
unset LD_LIBRARY_PATH NCCL_NET_PLUGIN
export LD_LIBRARY_PATH=/sw/user/gsl/lib
if [ -n "$_PLUG" ]; then
  export LD_LIBRARY_PATH=$_PLUG:/opt/cray/libfabric/2.3.1/lib64:$_SP/cuda_runtime/lib:$_SP/cuda_cupti/lib:$LD_LIBRARY_PATH
  export FI_PROVIDER=cxi NCCL_NET_OFI_PROVIDER=cxi FI_MR_CACHE_MONITOR=userfaultfd FI_CXI_DISABLE_HOST_REGISTER=1 \
         FI_CXI_DEFAULT_CQ_SIZE=131072 FI_CXI_DEFAULT_TX_SIZE=2048 FI_CXI_RDZV_EAGER_SIZE=0 FI_CXI_RDZV_PROTO=alt_read \
         FI_CXI_RX_MATCH_MODE=hybrid NCCL_CROSS_NIC=1 NCCL_DMABUF_ENABLE=0 NCCL_IB_DISABLE=1 NCCL_NET_GDR_LEVEL=PHB \
         NCCL_PLUGIN_P2P=0 NCCL_SOCKET_IFNAME=hsn
fi
export OFI_PLUGIN
