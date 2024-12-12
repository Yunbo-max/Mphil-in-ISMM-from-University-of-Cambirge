def loadDataSet(fileName):

    numFeat = len(open(fileName).readline().split(',')) - 1
    xArr = []
    yArr = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        currLine = line.strip().split(',')
        currLine.pop(0)
        currLine.pop(2)
        currLine.pop(1)
        currLine.pop(-1)
        numFeat = len(currLine)
        for i in range(numFeat):
            lineArr.append(float(currLine[i]))
        xArr.append(lineArr)
        yArr.append(float(currLine[-1]))
    return xArr,yArr

xArr,yArr = loadDataSet('Data')
print(xArr[:2])