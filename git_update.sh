#
DATE=`date`
WHO=`whoami`

git pull
git add . && git commit -m "$WHO $DATE" && git push

