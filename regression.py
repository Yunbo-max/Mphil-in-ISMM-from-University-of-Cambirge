from numpy import *
import matplotlib.pyplot as plt
import numpy as np


def loadDataSet(fileName):
    """

    """
    numFeat = len(open(fileName).readline().split(',')) - 1
    xArr = []
    yArr = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        currLine = line.strip().split(',')
        currLine.pop(0)
        currLine.reverse()
        numFeat = len(currLine)
        # print(numFeat)
        print(currLine)
        for i in range(numFeat-1):
            lineArr.append(float(currLine[i]))
        xArr.append(lineArr)
        yArr.append(float(currLine[-1]))
    print(xArr)
    print(yArr)
    return xArr,yArr


def standRegres(xArr, yArr):
    """

    """
    xMat = mat(xArr)
    print(xMat)
    yMat = mat(yArr).T
    xTx = xMat.T * xMat
    print(linalg.det(xTx))

    if linalg.det(xTx) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return

    ws = xTx.I*(xMat.T*yMat) # n*1
    # ws = linalg.solve(xTx,xMat.T*yMat)
    return ws

def plotFigure(xArr,yArr,yHat):
    """

    """
    xMat = mat(xArr)
    yMat = mat(yArr)
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.scatter(xMat[:, 1].flatten().A[0], yMat.T[:, 0].flatten().A[0],s=2,c='red')

    srtInd = xMat[:,1].argsort(0)
    xSort = xMat[srtInd][:,0,:]
    ax.plot(xSort[:,1],yHat[srtInd])
    plt.show()


def lwlr(testPoint,xArr,yArr,k=1.0):
    """

    """
    xMat = mat(xArr)
    yMat = mat(yArr).T
    m = shape(xMat)[0]
    weights = mat(eye((m)))
    for j in range(m):
        diffMat = testPoint - xMat[j,:]
        weights[j,j] = exp(diffMat*diffMat.T/(-2*k**2))
    xTx = xMat.T * (weights * xMat)
    if linalg.det(xTx) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    ws = xTx.I*(xMat.T*(weights*yMat)) # n*1
    return testPoint * ws

def lwlrTest(testArr,xArr,yArr,k=1.0):
    """

    """
    m = shape(testArr)[0]
    yHat = zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i],xArr,yArr,k)
    return yHat

if __name__ == "__main__":
    xArr,yArr = loadDataSet('Data')
    ws = standRegres(xArr,yArr)
    print(ws)
    yHat = (xArr*ws).flatten().A[0]
    plotFigure(xArr, yArr, yHat)


    xArr, yArr = loadDataSet('Data')
    k = 0.0001
    ws = lwlr(xArr[0], xArr, yArr, k)
    print("k=" + str(k) + "时的回归系数：\n", ws)
    yHat = lwlrTest(xArr, xArr, yArr, k)
    plotFigure(xArr, yArr, yHat)