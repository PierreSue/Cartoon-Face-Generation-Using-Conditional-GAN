## Usage :
### Training
training the VAE model:
* `$ python3 main.py --dataroot /training/image/folder --outf /output/folder --cuda`
Example:
* `$ python3 main.py --dataroot ../selected_cartoonset100k --outf ./result/ --cuda`

### testing
testing the VAE model:
* `$ python3 test.py --outf /output/folder/ --netVAE /model/path --cuda`
Example:
* `$ python3 test.py --outf ./images/ --netVAE ./checkpoint/netVAE.pth  --cuda`
Generate result from trained model:
* `$ bash bonus.sh /output/folder/`
