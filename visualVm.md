

### 服务端

由于docker环境内只有jre，没有jdk因此无法使用visualVm，下载[华为JDK](http://jdk.rnd.huawei.com/)，注意操作系统版本

通过xftp上传到 `/home/cspexpert`\

linux系统中jdk上传到docker内 ，这里pcg所在的容器为vcnapi

```sh
docker cp /home/cspexpert/jdk-8u292-linux-x64.tar.gz `docker ps | grep vcnapi | awk '{print $1}'`:/home
```

进入docker容器

```sh
docker exec -it -u root  `docker ps |grep vcnapi |awk '{print $1}'` bash
```

拷贝到指定路径

```
cp /home/jdk-8u292-linux-x64.tar.gz /usr/lib/jvm
```

解压

```
tar -zxvf jdk-8u292-linux-x64.tar.gz 
```

修改权限

```
chmod 777 -R ./jdk1.8.0_292/
```

修改环境变量

```
 vi /etc/profile
```

修改JAVA_HOME

```
export JAVA_HOME=/usr/lib/jvm/jdk1.8.0_292/jre
```

程序启动的shell脚本增加配置，目的是开启JMX 以便使用visualVm进行远程监控

```shell
-Dcom.sun.management.jmxremote.port=36666 -Dcom.sun.management.jmxremote.rmi.port=36666  -Dcom.sun.management.jmxremote.ssl=false -Dcom.sun.management.jmxremote.authenticate=false -Djava.rmi.server.hostname=90.71.166.31
```

重启

```
ps -ef | grep ivs_pcg | grep -v grep | awk '{print $2}' | xargs kill -9
```

### 客户端

下载[visualVm](https://visualvm.github.io/download.html)客户端

修改配置文件 `visualvm_21\etcvisualvm.conf` ,`visualvm_jdkhome="C:\Program Files\Huawei\jdk1.8.0_272"`  ,JDK指向本机JDK目录

Remote ->Add Remote Host->Add JMX Connection,添加java启动项里面配置的IP和port
