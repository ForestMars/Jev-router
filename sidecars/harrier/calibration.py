# from tier3_server import calibration as cal_mod
# cal = cal_mod.IdentityCalibration()

class IdentityCalibration:
    def map(self, raw):
        return raw