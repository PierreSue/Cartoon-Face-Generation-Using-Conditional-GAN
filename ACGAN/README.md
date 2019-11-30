## Usage:
### Training
training the gan model:
* `$ python3 main.py --dataroot /training/image/folder --outf /output/folder --cuda`
Example:
* `$ python3 main.py --dataroot ../selected_cartoonset100k --outf ./result/ --cuda`

### testing
testing the gan model:
* `$ python3 test.py --testroot /testing/label/file --outf /output/folder/ --netG /model/path --cuda`
Example:
* `$ python3 test.py --testroot ../sample_test/sample_fid_testing_labels.txt --outf ./images/ --netG ./checkpoint/netG.pth  --cuda`
Generate result from trained model:
* `$ bash cgan.sh /testing/label/file /output/folder/`
