import tensorflow as tf 

print("If its 1 then you have CUDDA. If it is 0 or something else you do not have it installed")
print("Number of GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
print("")

print("Another way to check is with this:")
print("If your name of your GPU shows up then you have it installed otherwise you don't have it installed")
print("")

if tf.test.gpu_device_name(): 
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
   print("Please install CUDDA")

####### SMALL DESCRIPTION ########
#If you have installed those checks will work for you and the print statments are self explanitory 
#If not check some tutorials online on how to install it 