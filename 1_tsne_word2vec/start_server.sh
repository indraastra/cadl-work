python -m server --train_data=embedding/text8 --eval_data=embedding/questions-words.txt --save_path=embedding/checkpoints
sudo redir --lport=80 --laddr=0.0.0.0 --cport 5000 --caddr=127.0.0.1
