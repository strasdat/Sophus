#!/usr/bin/env python3
from pathlib import Path

from setuptools import Command
from setuptools import setup
from setuptools.command.develop import develop
from setuptools.command.egg_info import egg_info
from setuptools.command.install import install

import shutil


class BuildProtosCommand(Command):
    user_options = []  # type: ignore

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        from grpc_tools import command

        proto_files_root = Path("protos")
        command.build_package_protos(proto_files_root)

        for proto_def in proto_files_root.rglob("*.proto"):
            proto_def_new = Path("py", *proto_def.parts[1:])
            shutil.copy(proto_def, proto_def_new)

        for proto_file in proto_files_root.rglob("*_pb2*.py"):
            proto_file_new = Path("py", *proto_file.parts[1:])
            if not proto_file_new.exists():
                proto_file.rename(proto_file_new)
            if proto_file.exists():
                proto_file.unlink()
        for proto_file in proto_files_root.rglob("*_pb2*.pyi"):
            proto_file_new = Path("py", *proto_file.parts[1:])
            if not proto_file_new.exists():
                proto_file.rename(proto_file_new)
            if proto_file.exists():
                proto_file.unlink()


class CleanFilesCommand(Command):
    user_options = []  # type: ignore

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        proto_files_root = Path("py/farm_ng")
        for proto_def in proto_files_root.rglob("*.proto"):
            assert proto_def.unlink() is None
        for proto_file in proto_files_root.rglob("*_pb2*.py"):
            assert proto_file.unlink() is None
        for proto_file in proto_files_root.rglob("*_pb2*.pyi"):
            assert proto_file.unlink() is None


class BuildProtosInstall(install):
    def run(self):
        # 1. Build the protobufs
        BuildProtosCommand.run(self)
        # 2. Run the installation
        install.run(self)
        # 3. Clean the generated protobufs
        CleanFilesCommand.run(self)


class BuildProtosDevelop(develop):
    def run(self):
        # 1. Build the protobufs
        BuildProtosCommand.run(self)
        # 2. Run the installation
        develop.run(self)


class BuildProtosEggInfo(egg_info):
    def run(self):
        # 1. Build the protobufs
        BuildProtosCommand.run(self)
        # 2. Run the installation
        egg_info.run(self)


setup(
    cmdclass={
        "build_package_protos": BuildProtosCommand,
        "install": BuildProtosInstall,
        "develop": BuildProtosDevelop,
        "egg_info": BuildProtosEggInfo,
        "clean": CleanFilesCommand,
    },
)
