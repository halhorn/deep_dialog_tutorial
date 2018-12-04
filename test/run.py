from unittest import TestLoader
from unittest import TextTestRunner
import sys
import os


def run(path=None):
    '''
    テストを実行します。

    :param str path: 指定された場合、そのパスに対応するテストを実行します。
    '''

    project_dir = './'

    if path:
        tests = _get_tests_from_file_path(path, project_dir)
    else:
        tests = TestLoader().discover(
            os.path.join(project_dir, 'test/'),
            pattern='*.py',
            top_level_dir=project_dir
        )

    return_code = not TextTestRunner().run(tests).wasSuccessful()
    sys.exit(return_code)


def _get_tests_from_file_path(path, project_dir):
    if not path.endswith('.py'):
        raise Exception('test file path should not dir path')

    # path は test/hoge/fuga.py などで与えられる
    path = os.path.relpath(path, project_dir)
    if not path.startswith('test/'):
        path = 'test/' + path

    # test.hoge.fuga に変換
    module_name = path.replace('.py', '').replace('/', '.')
    return TestLoader().loadTestsFromName(module_name)


if __name__ == '__main__':
    run(*sys.argv[1:])
