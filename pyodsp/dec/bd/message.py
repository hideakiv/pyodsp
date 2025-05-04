from ..run._message import InitMessage, FinalMessage, DnMessage, UpMessage


class BdInitMessage(InitMessage):
    pass


class BdUpMessage(UpMessage):
    pass


class BdDnMessage(DnMessage):
    pass


class BdFinalMessage(FinalMessage):
    pass
