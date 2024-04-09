class PID:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.error_sum = 0
        self.last_error = 0

    def update(self, setpoint, process_variable, dt):
        error = setpoint - process_variable
        self.error_sum += error * dt
        error_diff = (error - self.last_error) / dt

        output = self.kp * error + self.ki * self.error_sum + self.kd * error_diff

        self.last_error = error

        return output
    
    def reset(self):
        self.error_sum = 0
        self.last_error = 0
    