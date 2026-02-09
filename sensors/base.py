class LaneSensor:
    def get_metrics(self):
        """
        Returns a dict with:
        queue   : int
        waiting : float
        speed   : float
        """
        raise NotImplementedError
