import subprocess

subprocess.run(
    "pybind11-stubgen sophus_pybind -o sophus_pybind-stubs/",
    shell=True,
    check=True,
)

subprocess.run("touch sophus_pybind-stubs/py.typed", shell=True, check=True)
