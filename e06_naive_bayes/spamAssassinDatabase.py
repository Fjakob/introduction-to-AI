import re
import os
import random
from collections import namedtuple

MailData = namedtuple('MailData', ['label', 'words'])


class SpamAssassinDatabase(object):
    mail_pattern = re.compile('[0-9]{4}\.[a-z0-9]{32}')

    def __init__(self, data_path, training_test_ratio=.9):
        """ Load all mails in the data directory, setup stopwords if possible

        :param data_path: Path in which the spamAssasin datafiles are stored
        :param training_test_ratio: Ratio of datapoints assigned to training
                                    dataset and test dataset respectively
        """
        self.mailpaths = self._get_spamassasin_filepaths(data_path)
        self.training_test_ratio = training_test_ratio

    def read_training_mails(self):
        """ Get the datapoints belonging to training dataset

        :returns: Generator yielding MailData instances
        """
        return self._read_mails(0, self.training_test_ratio)

    def read_test_mails(self):
        """ Get the datapoints belonging to test dataset

        :returns: Generator yielding MailData instances
        """
        return self._read_mails(self.training_test_ratio, 1)

    def _read_mails(self, start, end):
        """ Get the datapoints from start index to end index

        :param start: Relative start offset
        :param end: Relative end index
        :returns: Generator yielding MailData instances
        """
        # yield every mail
        for klass in ['spam', 'ham']:
            mailpaths = self.mailpaths[klass]
            length = len(mailpaths)
            mailpaths = mailpaths[int(start * length):int(end * length)]
            for filepath in mailpaths:
                with open(filepath, 'rU') as f:
                    words = self._extract_mail_words(f.read())
                    yield MailData(label=klass, words=words)

    def _extract_mail_words(self, raw_content):
        """ Extract the relevant words from the raw mail content. I.e. remove
        headers, strip stop and illegal words, stem words.

        :param raw_content: The raw email content
        :returns: A clean list of words from the email's body
        """
        content = self._get_body(raw_content)
        content = self._remove_punctuation(content)
        content = content.lower()
        words = content.split(' ')

        # remove empty words
        words = filter(lambda x: x, words)

        # remove words with less than 3 characters
        words = filter(lambda x: len(x) > 3, words)

        # remove words containing digits
        words = filter(lambda x: not re.search('[0-9]', x), words)

        return words

    def _get_spamassasin_filepaths(self, data_path):
        """ Get all SpamAssassin files in the data directory.

        :param data_path: Path to the data directory.
        :returns Dictionary containing a list of filepaths for ham and spam.
        """
        # read the filelist of all available mails
        mailpaths = {'ham': [], 'spam': []}
        for root, dirs, files in os.walk(data_path):
            filenames = filter(lambda x: self.mail_pattern.match(x), files)
            filenames = map(lambda x: '/'.join((root, x)), files)
            label = 'spam' if 'spam' in root else 'ham'
            mailpaths[label] += filenames
        # Shuffle the paths for each class
        random.shuffle(mailpaths['ham'])
        random.shuffle(mailpaths['spam'])
        return mailpaths

    def _get_body(self, raw_content):
        """ Get body of an email, i.e. strip the headers.

        :param raw_content: The unmodified email content.
        :returns: The body of an email.
        """
        # remove email headers
        header_end_position = raw_content.find('\n\n')
        return raw_content[header_end_position:]

    def _remove_punctuation(self, content):
        """ Replace all characters except alpha numerical by whitespaces.

        :param content: String to clean.
        :returns: String containing alphanumerical characters only
        """
        return re.sub('[^a-zA-Z0-9]', ' ', content)