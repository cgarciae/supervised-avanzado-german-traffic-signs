import urllib
import zipfile
import os

TRAINING_SET_URL = "http://benchmark.ini.rub.de/Dataset/GTSRB_Final_Training_Images.zip"
TEST_SET_URL = "http://benchmark.ini.rub.de/Dataset/GTSRB_Final_Test_Images.zip"
TEST_CSV_URL = "http://benchmark.ini.rub.de/Dataset/GTSRB_Final_Test_GT.zip"


def get_progress():
    def progress(count, blockSize, totalSize):

        new = int( count * blockSize * 100 / totalSize )

        if new % 5 == 0 and new != progress.last:
            print("{}%".format(new))
            progress.last = new

    progress.last = -1

    return progress

if not os.path.exists("data"):
    os.makedirs("data")

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


print("downloading GT-final_test.csv.zip")
testfile = urllib.URLopener()
testfile.retrieve(TEST_CSV_URL, "data/GT-final_test.csv.zip", get_progress())


print("extracting GT-final_test.csv.zip")
zip_ref = zipfile.ZipFile("data/GT-final_test.csv.zip", 'r')
zip_ref.extract("GT-final_test.csv", "data/test-set/GTSRB/Final_Test/Images")
zip_ref.close()



print("removing files")
os.remove("data/training-set.zip")
os.remove("data/test-set.zip")
os.remove("data/GT-final_test.csv.zip")
os.remove("data/test-set/GTSRB/Final_Test/Images/GT-final_test.test.csv")


print("DONE")
