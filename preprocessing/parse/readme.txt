# First, download corenlp as instructed by 
# the version I use is stanford-corenlp-3.9.2, stanford-corenlp-full-2018-10-05.zip
# Unzip to jars folder corenlp/

nohup ./server.sh > sever.log &
./const_parse.sh
./dependency_parse.sh
kill xxx(server.sh pid)

# Done!

