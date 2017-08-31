from tkinter import *
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pickle
import numpy as np
import math
from multiprocessing import Pipe, Process
import _thread as thread
import sys,os
if sys.platform=='win32':
    import win32api,win32con
import time
import os
from tkinter.messagebox import *
from tkinter.filedialog import *
controldatalen=-1

def recvPipeData(pipe):
    data = b''
    while True:
        curdata = pipe.recv_bytes(1024)
        data += curdata
        if (not curdata) | (len(curdata) < 1024):
            break
    return data

def sendPipeControl(pipe,data):
    if isinstance(data, str):
        data=str.encode('utf-8')
    if controldatalen>0:
        if len(data)<controldatalen:
            data=data+b'_'*(controldatalen-len(data))
        else:
            data=data[:controldatalen]
    pipe.send_bytes(data)

def monitorPipe(pipe,praseFunc):
    while True:
        data = recvPipeData(pipe)
        if data:
            praseFunc(data)
        else:
            showInfo("null data")

def showInfo(msg):
    print(msg)

class LineShower(Frame):
    def __init__(self, parent, title=None, xlabel=None, ylabel=None,linenum=1, pipe=None,linelabellist=None, maxnum=-1):
        Frame.__init__(self, parent)
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.title = title
        self.linenum = linenum
        self.maxnum = maxnum
        self.linelabellist=linelabellist
        self.pipe = pipe
        self.data = []
        for i in range(self.linenum):
            self.data.append([])
        self.__initView()
        self.pack(expand=YES, fill=BOTH)
        self.bind()
        if self.pipe:
            thread.start_new_thread(monitorPipe,(self.pipe,self.praseFunc))

    def praseFunc(self,data):
        try:
            res = pickle.loads(data)
            self.addData(res)
        except pickle.UnpicklingError:
            showInfo("wrongData:UnpicklingError")

    def __initView(self):
        controlrow=Frame(self)
        controlrow.config(background='#ffffff')
        controlrow.pack(side=TOP,fill=X)
        savebt=Button(controlrow,text="保存数据")
        savebt.pack()
        savebt.config(command=self.saveData)
        figure = Figure(figsize=(7, 5), dpi=96)
        self.plot = figure.add_subplot(111)
        self.setViewInfo()
        self.canvas = FigureCanvasTkAgg(figure, self)
        self.canvas.get_tk_widget().pack(anchor=E, expand=YES, fill=BOTH)

    def saveData(self):
        existdata=False
        for item in self.data:
            if item:
                existdata=True
        if not existdata:
            if not askyesno("信息","无数据，是否继续？"):
                return
        file=asksaveasfile(defaultextension='.csv',filetypes=[('CSV(逗号分隔)','*.csv'),('文本','*.txt')])
        if file:
            file.write('"linelabel","linenum",%s,%s\n'%(self.xlabel,self.ylabel))
            for i,item in enumerate(self.data):
                for d in item:
                    file.write('%s,%s,%s,%s\n'%(self.linelabellist[i],i+1,d[0],d[1]))
            file.close()

    def setViewInfo(self):
        if self.title:
            self.plot.set_title(self.title)
        if self.xlabel:
            self.plot.set_xlabel(self.xlabel)
        if self.ylabel:
            self.plot.set_ylabel(self.ylabel)

    def getData(self):
        return self.data

    def clearData(self):
        datalist=[]
        for dataitem in self.data:
            datalist.append([])
        self.data=datalist
        self.rerendGraph()

    def addData(self, data):
        if data[0] >= self.linenum|data[0]<=0:
            self.showInfo("unknownData")
            return
        itemnum = data[0] - 1
        if self.maxnum > 0:
            while len(self.data[itemnum]) >= self.maxnum:
                self.data[itemnum].pop(0)
        self.data[itemnum].append(data[1:])
        self.rerendGraph()

    def rerendGraph(self):
        self.plot.clear()
        self.setViewInfo()
        haslabel=False
        for i,item in enumerate(self.data):
            if not item:
                self.plot.plot(np.array([0]), np.array([0]))
                continue
            haslabel=True
            itemarray = np.array(item)
            if self.linelabellist[i]:
                self.plot.plot(itemarray[:, 0], itemarray[:, 1], label=self.linelabellist[i])
            else:
                haslabel=False
                self.plot.plot(itemarray[:, 0], itemarray[:, 1])
        if haslabel:
            self.plot.legend(loc="upper center")
        self.canvas.draw()

class Shower(Frame):
    def __init__(self, root,showDefine, pipelist,controlpipe,showStartBt=True):
        """
        训练显示器
        :param showDefine:[(title,xlabel,ylabel,linenum,maxnum),(title,xlabel,ylabel,linenum,maxnum)...]
        :param trainFunc:训练的方程，管道作为输入。
        """
        Frame.__init__(self,root)
        self.vnum=len(showDefine)
        self.root=root
        col = int(math.sqrt(self.vnum))
        if math.pow(col,2) < self.vnum:
            col = col + 1
        self.viewlist = []
        self.pipe = pipelist
        self.controlpipe=controlpipe
        num = 0
        if isinstance(showDefine,ViewSetup):
            showDefine=showDefine.config
        while True:
            if num>=self.vnum:
                break
            row = Frame(self)
            row.pack(side=TOP, fill=BOTH, expand=YES)
            for i in range(col):
                if num >= self.vnum:
                    break
                view = LineShower(row, title=showDefine[num][0], xlabel=showDefine[num][1], ylabel=showDefine[num][2],
                                  linenum=showDefine[num][3], pipe=pipelist[num],linelabellist=showDefine[num][4], maxnum=showDefine[num][5])
                view.pack(side=LEFT, expand=YES, fill=BOTH)
                self.viewlist.append(view)
                num += 1
        if showStartBt:
            self.startbt=Button(self,text="开始")
            self.startbt.pack(side=TOP)
            self.startbt.config(command=self.startFunc)
        self.pack(fill=BOTH,expand=YES)
        thread.start_new_thread(monitorPipe,(self.controlpipe,self.praseFunc))

    def startFunc(self):
        try:
            self.startbt.config(state=DISABLED)
        except AttributeError:
            pass
        for view in self.viewlist:
            view.clearData()
        sendPipeControl(self.controlpipe,b'start')

    def praseFunc(self,data):
        if data==b'complete':
            self.onComplete()

    def onComplete(self):
        showinfo("信息！","后台程序运行完成！")
        try:
            self.startbt.config(state=NORMAL,text="重新开始")
        except AttributeError:
            pass

    def pipeFuncModule(self,pipe):
        def func(x,y,num=0):
            data=pickle.dumps((num,x,y))
            pipe.send_bytes(data)
        return func

    def destroy(self,backgroundexit=False):
        if backgroundexit:
            sendPipeControl(self.controlpipe,b'exit')

class Runer:
    def __init__(self,pipelist,controlpipe,func):
        self.runFunc=func
        self.pipelist=pipelist
        self.controlpipe=controlpipe

    def praseFunc(self,data):
        if data==b'start':
            thread.start_new_thread(self.startFunc,())
        elif data==b'exit':
            self.exit()

    def sendDataModule(self,pipe):
        def sendFunc(x,y,num=0):
            data=pickle.dumps((num,x,y))
            pipe.send_bytes(data)
        return  sendFunc

    def startloop(self):
        self.lock=thread.allocate_lock()
        self.continueloop=True
        thread.start_new_thread(monitorPipe,(self.controlpipe,self.praseFunc))
        while True:
            self.lock.acquire()
            if not self.continueloop:
                self.lock.release()
                break
            self.lock.release()
            time.sleep(0.5)

    def exit(self):
        self.lock.acquire()
        self.continueloop=False
        self.lock.release()

    def startFunc(self):
        sendfunclist=[]
        for pipe in self.pipelist:
            func=self.sendDataModule(pipe)
            sendfunclist.append(func)
        self.runFunc(*sendfunclist)
        sendPipeControl(self.controlpipe,b"complete")

class ViewSetup:
    def __init__(self):
        self.config=[]

    def addView(self,title=None, xlabel=None, ylabel=None,linenum=1,linelabellist=None, maxnum=-1):
        if not linelabellist:
            linelabellist=[]
            for i in range(linenum):
                linelabellist.append(None)
        if linenum!=len(linelabellist):
            raise ValueError("linelabellist's item number should equal to linenum")
        self.config.append((title,xlabel,ylabel,linenum,linelabellist,maxnum))

    def getConfig(self):
        return self.config

def winExit(root,s):
    def func():
        res=askyesnocancel("警告！","是否退出后关闭后台进程？")
        if res==True:
            s.destroy(True)
            root.destroy()
        elif res==False:
            root.destroy()
        elif res==None:
            pass
    return func

def runShower(config,pipelist,controlpipe,shower=Shower):
    root=Tk()
    s=shower(root,config,pipelist,controlpipe)
    root.protocol("WM_DELETE_WINDOW", winExit(root,s))
    root.mainloop()

def runFunc(runfunc,pipelist,controlpipe,runner=Runer):
    r=runner(pipelist,controlpipe,runfunc)
    r.startloop()

def start(config,func,shower=Shower,runner=Runer):
    if isinstance(config,ViewSetup):
        config=config.config
    parentpipelist = []
    childpipelist=[]
    for i in range(len(config)):
        parentp, chilep = Pipe()
        parentpipelist.append(parentp)
        childpipelist.append(chilep)
    pcontrol,ccontrol=Pipe()
    p=Process()
    p = Process(target=runFunc, args=(func,childpipelist,ccontrol,runner))
    p.start()
    runShower(config,parentpipelist,pcontrol,shower)
    p.join()

def testFunc(sendfunc1,sendfunc2):
    import time
    import random
    num=0
    for i in range(50):
        sendfunc1(num,random.randrange(0,10),1)
        sendfunc2(2*num,random.randrange(0,10),1)
        time.sleep(0.2)
        sendfunc1(num,random.randrange(0,10),2)
        sendfunc2(2*num+1,random.randrange(0,10),1)
        time.sleep(0.2)
        num+=1

if __name__=='__main__':
    config=ViewSetup()
    config.addView(linenum=2)
    config.addView()
    start(config, testFunc)