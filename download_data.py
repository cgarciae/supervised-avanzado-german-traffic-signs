import urllib
import zipfile

TRAINING_SET_URL = "http://benchmark.ini.rub.de/Dataset/GTSRB_Final_Training_Images.zip"
TEST_SET_URL = "http://benchmark.ini.rub.de/Dataset/GTSRB_Final_Test_Images.zip"


def get_progress():
    def progress(count, blockSize, totalSize):

        new = int( count * blockSize * 100 / totalSize )

        if new % 5 == 0 and new != progress.last:
            print("{}%".format(new))
            progress.last = new

    progress.last = -1

    return progress

print("downloading training-set.zip")
testfile = urllib.URLopener()
testfile.retrieve(TRAINING_SET_URL, "data/training-set.zip", get_progress())

print("downloading test-set.zip")
testfile = urllib.URLopener()
testfile.retrieve(TEST_SET_URL, "data/test-set.zip", get_progress())


print("extracting training-set.zip")
zip_ref = zipfile.ZipFile("data/training-set.zip", 'r')
zip_ref.extractall("data/training-set")
zip_ref.close()


print("extracting test-set.zip")
zip_ref = zipfile.ZipFile("data/test-set.zip", 'r')
zip_ref.extractall("data/test-set")
zip_ref.close()




