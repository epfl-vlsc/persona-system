
rm /tmp/tensorflow_pkg/*.whl
bazel build -c opt //tensorflow/tools/pip_package:build_pip_package
bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
sudo pip uninstall tensorflow
sudo -H pip install /tmp/tensorflow_pkg/tensorflow-*.whl --upgrade

