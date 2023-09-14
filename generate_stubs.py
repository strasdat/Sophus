import subprocess

subprocess.run("mkdir -p sophus_pybind-stubs/sophus_pybind", shell=True, check=True)

subprocess.run(
    "pybind11-stubgen sophus_pybind -o sophus_pybind-stubs/sophus_pybind",
    shell=True,
    check=True,
)
