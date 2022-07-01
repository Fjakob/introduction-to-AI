"""
Implement a naive Bayes classificator for spam emails.

We supply a database of spam/ham emails as well as a small library to read it.
The database is in the subfolder database. All spam mails are in
database/spam_mails, all ham emails in database/ham_emails.

To load the database write

    db = spamAssassinDatabase.SpamAssassinDatabase(data_path='./database',
                                                   training_test_ratio=.75)

The data_path parameter gives the path to the database, the training_test_ratio
defines which part of the database should be read as training data and which
part should be used as test data.

You can then access the data with:

    word_count_spam = {}
    word_count_ham = {}

    for mail in db.read_training_mails():
        for w in mail.words:
             if mail.label == 'spam':
                 word_count_spam[w] += 1
             elif mail.label == 'ham':
                 word_count_ham[w] += 1

You can use the word frequency to estimate the probabilities p(x|c).
"""
import spamAssassinDatabase
import random


class NaiveBayes(object):
    
    def __init__(self, database):
        ''' Train the classificator with the given database. '''
        a, b, c, d = self.estimate_prob(database)
        self.prob_given_spam = a
        self.prob_given_ham = b
        self.prob_spam = c
        self.prob_ham = d
        
    def estimate_prob(self, database):
        
        word_count_spam = {}
        word_count_ham = {}
        hams = 0.
        spams = 0.
    
        for mail in database.read_training_mails():
            if mail.label == 'spam':
                spams += 1.
            elif mail.label == 'ham':
                hams += 1.
            for word in mail.words:
                 if mail.label == 'spam':
                     try:
                         word_count_spam[word] += 1
                     except(KeyError):
                         word_count_spam[word] = 1
                 elif mail.label == 'ham':
                     try:
                         word_count_ham[word] += 1
                     except(KeyError):
                         word_count_ham[word] = 1
                         
        prob_ham = hams / (hams + spams)
        prob_spam = 1. - prob_ham
        prob_given_spam = {}
        prob_given_ham = {}  
           
        for word in word_count_spam:
            spam_count = float(word_count_spam[word])
            try:
                ham_count = float(word_count_ham[word])
            except(KeyError):
                ham_count = 0.
            prob_given_spam[word] = spam_count / (spam_count + ham_count)
            
        for word in word_count_ham:
            ham_count = float(word_count_ham[word])
            try:
                spam_count = float(word_count_spam[word])
            except(KeyError):
                spam_count = 0.
            prob_given_ham[word] = ham_count / (ham_count + spam_count)
        
        return prob_given_spam, prob_given_ham, prob_spam, prob_ham


    def spam_prob(self, email):
        ''' Compute the probability for the given email that it is spam. '''
        
        joint_spam = self.prob_spam
        joint_ham = self.prob_ham
        
        for word in email.words:
            if word in self.prob_given_spam:
                joint_spam *= self.prob_given_spam[word]
                if word not in self.prob_given_ham:
                    joint_ham *= 0.0000005      
            if word in self.prob_given_ham:
                joint_ham *= self.prob_given_ham[word]
                if word not in self.prob_given_spam:
                    joint_spam *= 0.0000005
        
        try:        
            prob_spam_given_words = joint_spam / (joint_ham + joint_spam)
        except(ZeroDivisionError):
            return 1
        
        return prob_spam_given_words


def main():
    db = spamAssassinDatabase.SpamAssassinDatabase(data_path='./database',
                                                   training_test_ratio=.75)

    nb = NaiveBayes(db)
    correctc = 0.
    falsec = 0.

    for n, mail in enumerate(db.read_test_mails()):
        prob = nb.spam_prob(mail)
        correct = ((prob > .5 and mail.label == 'spam')
                   or (prob <= .5 and mail.label == 'ham'))
        if correct: correctc += 1
        else: falsec += 1
        print("Mail {} -- p(c|x) = {} -- is {} -- Labeling correct: {}"
              .format(n, prob, mail.label, correct))

    print "\n{}% identified correctly".format((correctc/(correctc + falsec)))


if __name__ == '__main__':
    main()
