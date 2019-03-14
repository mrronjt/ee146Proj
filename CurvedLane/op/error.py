import numpy as np
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

with open("ErrorOP_CU4.txt", "r") as fp:
    for i, line in enumerate(fp):
        errors = line.split(' ')

#print(errors)
#print(len(errors))

leftlen = 0
rightlen = 0

for i in range(len(errors)):
    if errors[i] == '':
        continue
    if float(errors[i]) > 0.0:
        if i % 2 == 0: #even
            leftlen = leftlen + 1
        else:
            rightlen = rightlen + 1

lefterrors = np.zeros(leftlen)
righterrors = np.zeros(rightlen)
leftindex = 0
rightindex = 0

#print(len(errors))

for i in range(len(errors)):
    #print(float(errors[i]))
    if errors[i] == '':
        continue
    if float(errors[i]) > 0.0:
        if i % 2 == 0: #even
            lefterrors[leftindex] = errors[i]
            leftindex = leftindex + 1
        else:
            righterrors[rightindex] = errors[i]
            rightindex = rightindex + 1

#print(lefterrors)
#print(righterrors)

lefttime = np.arange(1,(len(lefterrors)+1),1)
righttime = np.arange(1,(len(righterrors)+1),1)
line1 = plt.plot(lefttime, lefterrors, "r,-", label="Left Line Error")
line2 = plt.plot(righttime, righterrors, "b,-", label="Right Line Error")
plt.title("PT: CU4 RMS Error")
plt.xlabel("Frames")
plt.ylabel("Error")

red_patch = mpatches.Patch(color="red", label="Left Line Error")
blue_patch = mpatches.Patch(color="blue", label="Right Line Error")
plt.legend(handles=[red_patch,blue_patch])

plt.show()
