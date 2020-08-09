import matplotlib.pyplot as plt

epoch=[]
gen_loss=[]
dis_loss=[]

log=open('log.txt','r').readlines()
for line in log:
    if line[0]=='E':
        a = line.split(' ')
        epoch.append(int(a[1].replace('\n','')))
    elif line[0]=='D':
        a = line.split(' : ')
        dis_loss.append(float(a[1].replace('\n','')))
    elif line[0]=='G':
        a = line.split(' : ')
        gen_loss.append(float(a[1].replace('\n','')))
print(gen_loss)
plt.plot(epoch,dis_loss)
plt.plot(epoch,gen_loss)
plt.show()
