#!/usr/bin/env python3
from setuptools import setup
from setuptools import Command
from setuptools.command.develop import develop
from setuptools.command.egg_info import egg_info
from setuptools.command.install import install
from pathlib import Path
from farm_ng.core.package import build_package_protos, clean_protos


class BuildProtosCommand(Command):
    user_options = []  # type: ignore

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        build_package_protos(proto_root=Path("protos"), package_root=Path("py"))


class CleanFilesCommand(Command):
    user_options = []  # type: ignore

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        clean_protos(package_root=Path("py"))


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
    }
)
