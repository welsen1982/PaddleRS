# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tempfile

from testing_utils import run_script

if __name__ == '__main__':
    with tempfile.TemporaryDirectory() as td:
        run_script(
            f"python match.py --image1_path ../tests/data/ssmt/multispectral_t1.tif --image2_path ../tests/data/ssmt/multispectral_t1.tif --save_path {td}/out.tiff",
            wd="../tools")
