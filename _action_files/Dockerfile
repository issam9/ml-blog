<<<<<<< HEAD
FROM fastai/jekyll:2020-09-12
=======
FROM fastai/jekyll:2020-12-30
>>>>>>> template/master

WORKDIR /fastpages
COPY . .
RUN chmod u+x action_entrypoint.sh
RUN chmod u+x word2post.sh
RUN dos2unix /fastpages/*.sh

CMD [ "/fastpages/action_entrypoint.sh" ]
