# Imports PIL module
from PIL import Image
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
# open method used to open different extension image file
im1 = Image.open(r"F:\Deepika\Paper_works\Sundas Almas (237915) - Paper 1 (Class I)\237915\Result\mse_tr.jpg")
im2 = Image.open(r"F:\Deepika\Paper_works\Sundas Almas (237915) - Paper 1 (Class I)\237915\Result\rmse_tr.jpg")
im3 = Image.open(r"F:\Deepika\Paper_works\Sundas Almas (237915) - Paper 1 (Class I)\237915\Result\r_sq_tr.jpg")
im4 = Image.open(r"F:\Deepika\Paper_works\Sundas Almas (237915) - Paper 1 (Class I)\237915\Result\mae_tr.jpg")
im5 = Image.open(r"F:\Deepika\Paper_works\Sundas Almas (237915) - Paper 1 (Class I)\237915\Result\mse_k.jpg")
im6 = Image.open(r"F:\Deepika\Paper_works\Sundas Almas (237915) - Paper 1 (Class I)\237915\Result\rmse_k.jpg")
im7 = Image.open(r"F:\Deepika\Paper_works\Sundas Almas (237915) - Paper 1 (Class I)\237915\Result\r_sq_k.jpg")
im8 = Image.open(r"F:\Deepika\Paper_works\Sundas Almas (237915) - Paper 1 (Class I)\237915\Result\mae_k.jpg")
im9 = Image.open(r"F:\Deepika\Paper_works\Sundas Almas (237915) - Paper 1 (Class I)\237915\Result\mse_perf.jpg")
im10 = Image.open(r"F:\Deepika\Paper_works\Sundas Almas (237915) - Paper 1 (Class I)\237915\Result\rmse_perf.jpg")
im11 = Image.open(r"F:\Deepika\Paper_works\Sundas Almas (237915) - Paper 1 (Class I)\237915\Result\r_sq_perf.jpg")
im12 = Image.open(r"F:\Deepika\Paper_works\Sundas Almas (237915) - Paper 1 (Class I)\237915\Result\mae_perf.jpg")



# This method will show image in any image viewer
im1.show()
im2.show()
im3.show()
im4.show()
im5.show()
im6.show()
im7.show()
im8.show()
im9.show()
im10.show()
im11.show()
im12.show()
