#!/bin/bash
export JAVA_HOME=/usr/java/jdk1.7.0_67
EXECPATH=$(dirname "$0")

function log()
{
    date_=`date +"%Y%m%d %H:%M:%S"`
    echo "${date_} $*"
}

if [[ $# -eq 1 ]];then
    DAY=$1
else
    DAY=`date -d "-1 day" "+%Y%m%d"`
fi
echo ${DAY}

# Hadoop Streaming 环境配置
VERSION=2.6.3
HPHOME_ROOT=/software/servers/hadoop-${VERSION}-bin/hadoop-${VERSION}
HPHOME=${HPHOME_ROOT}/bin/
JAR_PACKEGE=${HPHOME_ROOT}/share/hadoop/tools/lib/hadoop-streaming-${VERSION}.jar

# 队列名
QUEUENAME="ven2"

# 固网、移动日志目录
FIX_LOG_SRC="/daas/bstl/dpifix"
MOBILE_LOG_SRC="/daas/bstl/dpiqixin"
# 日志分析结果存储目录, 不同的脚本中可以重新配置此变量的值
LOG_RES="/user/shujuzu/yxc_tst_out/retarget"

# 各分步reduce任务个数设置
USER_PARSE_RDC_NUM=90
GET_QUICK_RDC_NUM=90
MATCH_QUICK_RDC_NUM=90


function run_fix_or_mobile()
{
    # $1 为 fix 或 mobile
    if [[ ! $1 =~ ^(fix|mobile)$ ]]; then
        log "ERROR: use function run_fix_or_mobile with fix or mobile as \$1"
        exit 1
    fi

    log "run_$1"

    if [ "$1" = "fix" ]; then
        # 指定输入路径
        IN_PATH=${FIX_LOG_SRC}/${province_fix}/${DAY}/${HOUR}
    else
        # 指定3G输入目录
        IN_PATH=${MOBILE_LOG_SRC}/${province_mobile}/${DAY}/${HOUR}
        # 指定4G输入目录
        IN_PATH_4G=${MOBILE_LOG_SRC}/${province_mobile_4G}/${DAY}/${HOUR}
    fi

    # 指定输出目录
    OUT_PATH=${LOG_RES}/${province}/${1}/${F_TAG}

    log "the province is ${province}, the parse job is ${DAY}/${HOUR} $1"

    ${HPHOME}hadoop fs -test -e ${OUT_PATH}
    if [ $? ]; then
        ${HPHOME}hadoop fs -rm -r -skipTrash ${OUT_PATH}
    fi

    echo "IN_PATH(fix|3G): ${IN_PATH}"
    echo "IN_PATH_4G: ${IN_PATH_4G}"
    echo "OUT_PATH: ${OUT_PATH}"

    # 指定Mapper模块
    MAP=${EXECPATH}/map_${1}.py
    # 指定Reducer模块
    RED=${EXECPATH}/reduce.py

    ${HPHOME}hadoop jar ${JAR_PACKEGE} \
        -D mapreduce.job.queuename=${QUEUENAME} \
        -D stream.non.zero.exit.is.failure=false \
        -D mapreduce.job.split.metainfo.maxsize=100000000 \
        -D mapreduce.reduce.failures.maxpercent=60 \
        -D mapreduce.map.failures.maxpercent=60 \
        -D mapreduce.job.reduces=${USER_PARSE_RDC_NUM} \
        -D mapreduce.output.fileoutputformat.compress=true \
        -D mapreduce.output.fileoutputformat.compress.codec=com.hadoop.compression.lzo.LzoCodec \
        -input ${IN_PATH} ${IN_PATH_4G} \
        -output ${OUT_PATH} \
        -inputformat SequenceFileAsTextInputFormat \
        -mapper ${MAP} \
        -reducer ${RED} \
        -file ${EXECPATH}/*
#       -cmdenv SEP=$1

    log "parse job : $1 is over"
}


# 跑一个省的第一阶段信息提取任务：
# 参数1： 省名，如zhejiang
# 参数2，3，4 如果该省的固网和移动网的目录名不一致，则需要用4个参数的方式，
# 一一指定【省名、固网目录名、3G移动网目录名、4G移动网目录名】
function run_province()
{
    province=$1
    province_fix=""
    province_mobile=""
    province_mobile_4G=""
    if [ $# -eq 1 ]; then
        province_fix=$1
        province_mobile=$1
        province_mobile_4G=$1"_4G"
    elif [ $# -eq 4 ]; then
        province_fix=$2
        province_mobile=$3
        province_mobile_4G=$4
    fi

    if [[ ${NET_T} == "all" ]]; then
        # 启动固网任务，参见对应的function
        run_fix_or_mobile fix
        # 启动移动网任务
        run_fix_or_mobile mobile
    else
        run_fix_or_mobile ${NET_T}
    fi
}

# - - - - - - - - - - - main - - - - - - - - - - - - - #
# 以下为程序开始执行的部分

# 判断该脚本是否拥有3个参数，如果有，则将其作为报文分析的日期、时间等参数
if [[ $# -eq 3 ]]; then
    DAY=$1
    F_TAG=${1%\*}${2%\*}
    HOUR=*${F_TAG}*
    NET_T=$3
else
    if [[ $# -eq 1 ]]; then
        DAY=$1
    else
        DAY=`date -d "-1 day" +"%Y%m%d"`
    fi
    F_TAG=${DAY}
    HOUR=*${F_TAG}*
    NET_T="all"
fi

log "Totally \" ${DAY} - ${HOUR} ${NET_T} \" start!!"

log "Totally \" ${DAY} - ${HOUR} ${NET_T} \" start!!"

# 按照省份列表依次执行分析工作。
provinces=(
#    anhui
#    beijing
#    chongqing
#    fujian
#    gansu
#    guangdong
#    guangxi
#    guizhou
#    hainan
#    hebei
#    "heilongjiang heilongjiang hlj heilongjiang_4G"
#    henan
#    hubei
#    hunan
#    jiangsu
#    jiangxi
#    jilin
#    liaoning
#    "neimenggu neimenggu nmg neimenggu_4G"
#    ningxia
#    qinghai
#    shandong
    shanghai
#    shanxi
#    shanxisheng
#    sichuan
#    tianjin
#    xinjiang
#    xizang
#    yunnan
#    zhejiang
)

for i in ${!provinces[*]}
do
    run_province ${provinces[$i]}
done

log "Totally \" ${DAY} - ${HOUR} ${NET_T} \" done!!"
exit 0
