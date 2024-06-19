#!/bin/bash

echo $HOME
echo $PWD
echo $SHELL
echo $USER

A=100
echo "The value of A is $A"
# readonly B=2

unset A
echo "The value of A is $A"
# unset B

A=`date`

echo "Date is $A"

A=$(echo "Hello")

echo "The value of A is $A"

# 同一个shell启动的其他进程可见
export ONE=1

# 第0参数，所有参数（整体），所有参数（分别），参数个数 (不包括 sh, ./learn.sh 本可执行文件参数)
echo "${0} , '$*' , '$@' , $#"


# 当前进程号，后台运行的最后一个进程号，上次命令是否失败（0为正确）
echo "$$ , $!, $?"

echo "value=$[3+2*2-1/1]"

T="5%2==1 value is $((5%2==1))"
echo "$T"

echo $(expr $(expr 3 \* 2) + 3)

echo $[$1+$2]

# 条件判断语句 [_ ... _] need space
# 短路执行的原理
[ -f "lea.sh" ] && echo OK2 || echo NOTOK

#  if with fi pair
FILE_NAME="learn"
#if [ -f $FILE_NAME ] || [ -d "logs" ]
if [ -f $FILE_NAME ] && [ -d "logs" ]
then
    echo "$FILE_NAME exists."
elif [ -f "$FILE_NAME.sh" ]
then 
    echo "$FILE_NAME.sh exists."
fi


# 字符串判断
OK="ok"
[ $OK = ok ]


echo $[$?==0]
echo $(($?==0))

# 常用判断条件 
#     1) = 字符串比较 
#     2) 两个整数的比较 
#       -lt 小于 
#       -le 小于等于 
#       -eq 等于 
#       -gt 大于 
#       -ge 大于等于 
#       -ne 不等于
#     3) 按照文件权限进行判断 
#       -r 有读的权限
#       -w 有写的权限 
#       -x 有执行的权限
#     4) 按照文件类型进行判断 
#       -f 文件存在并且是一个常规的文件 
#       -e 文件存在 
#       -d 文件存在并是一个目录


case $1 in
"1")

echo "$1 is 1"
echo "first branch"

;;

*)
echo "$1 is not 1"
echo "second branch"
;;

esac

[ -f "learn.sh" ] && [ -w "learn.sh" ]

echo $?


# 这里打印整体，但是如果没有引号的话，就分别打印了
for var in "$*"
do
    echo "_$var"
done


# 这里有没有引号都是分别打印
for var in "$@"
do
    echo "_$var"
done

# 列表添加
for var in 2 3 4 5 
do
    echo $[$var+1]

done

SUM=0
for (( i=1 ; i<=$1 ; i += 1 ))
do
    SUM=$[$SUM+$i]
done

echo SUM=$SUM


i=1
SUM_1=10  # 假设这是动态生成的变量
SUM=5     # 初始值
SUM_VAR="SUM_$i"

# 使用!将字符串变为变量名
SUM=$((SUM + ${!SUM_VAR}))
echo $SUM  # 输出 15

echo "$SUM_VAR"

echo "${!SUM_VAR}"


A="rere"
B=$A

B=3

echo $A$B


i=0
SUM=0
while [ ! $i -gt $2 ]
do
    SUM=$SUM$i
    echo $SUM
    echo test!
    # 这里不能写 i++
    i=$[$i+1]
done


read -t 10 -p "请输入一个数 NUM2=" NUM2 
echo "你输入的 NUM2=$NUM2"


basename ./stable-baselines3/CITATION.bib


dirname ./stable-baselines3/CITATION.bib



function test_func() {

    echo "test func implemented"
    return $[$2+$1]
}

test_func 3 4

echo $?