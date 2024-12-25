import random
import numpy as np
from scipy import stats
import json


def sample_truncated_normal(mean, std, lower, upper, size=1):
    a = (lower - mean) / std
    b = (upper - mean) / std
    return stats.truncnorm.rvs(a, b, loc=mean, scale=std, size=size)


def sample_truncated_exponential(lambda_param, a, b, size=1):
    p_a = 1 - np.exp(-lambda_param * a)
    p_b = 1 - np.exp(-lambda_param * b)
    u = np.random.uniform(p_a, p_b, size)
    samples = -np.log(1 - u) / lambda_param
    return samples


class Operation:
    def __init__(self, available_machine: list[int], operation_times: list[int]):
        self.available_machine = available_machine
        self.operation_times = operation_times

    def to_json(self):
        return {'available_machine': self.available_machine,
                'operation_times': self.operation_times}

    @staticmethod
    def generate_operation(num_machine, lower_pres_mu, upper_pres_mu,
                           sigma_pres_time, lower_pres_time, upper_pres_time,
                           mu_num_cap_mac, sigma_num_cap_mac, lower_num_cap_mac, upper_num_cap_mac):
        # num of machine capable
        val = int(
            sample_truncated_normal(mu_num_cap_mac, sigma_num_cap_mac, lower_num_cap_mac, upper_num_cap_mac).item())
        available_machine = sorted(random.sample(range(0, num_machine), val))  # sample val times from [0,num_machine)
        mu_pres_time = np.random.uniform(low=lower_pres_mu, high=upper_pres_mu, size=None)
        operation_times = sample_truncated_normal(mu_pres_time, sigma_pres_time, lower_pres_time, upper_pres_time,
                                                  size=val)
        operation_times = [int(t.item()) for t in operation_times]
        # print(available_machine, operation_times)
        operation = Operation(available_machine, operation_times)
        return operation


class Job:
    def __init__(self, operation_list: list[Operation]):
        self.operation_list = operation_list
        self.finish_time_min = sum([min(op.operation_times) for op in self.operation_list])
        self.num_operation = len(self.operation_list)

    def to_json(self):
        return [op.to_json() for op in self.operation_list]

    @staticmethod
    def generate_job(num_machine, num_operation, lower_pres_mu,
                     upper_pres_mu, sigma_pres_time, lower_pres_time, upper_pres_time,
                     mu_num_cap_mac, sigma_num_cap_mac, lower_num_cap_mac, upper_num_cap_mac
                     ):
        operation_list = [Operation.generate_operation(num_machine, lower_pres_mu, upper_pres_mu,
                                                       sigma_pres_time, lower_pres_time, upper_pres_time,
                                                       mu_num_cap_mac, sigma_num_cap_mac, lower_num_cap_mac,
                                                       upper_num_cap_mac) for _ in
                          range(num_operation)]
        job = Job(operation_list)
        return job


class Case:
    def __init__(self, init_job_list, suc_job_list, broken_down_list, num_machine):
        self.init_job_list = init_job_list
        self.suc_job_list = suc_job_list
        self.broken_down_list = broken_down_list
        self.num_machine = num_machine
        self.num_operation = sum([job.num_operation for job in self.init_job_list])
        self.num_operation_suc = sum([job.num_operation for job, ar_time in self.suc_job_list])

    def to_json(self):
        return {"init_job_list": [job.to_json() for job in self.init_job_list],
                "suc_job_list": [(job.to_json(), time) for job, time in self.suc_job_list],
                "broken_down_list": self.broken_down_list,
                "num_machine": self.num_machine}

    @staticmethod
    def generate_case(num_machine,
                      num_operation_max,
                      num_job,
                      dynamic_prop,
                      max_broken_down,
                      lower_num_opj,
                      upper_num_opj,
                      lower_pres_mu,
                      upper_pres_mu,
                      sigma_pres_time,
                      lower_pres_time,
                      upper_pres_time,
                      mu_num_cap_mac,
                      sigma_num_cap_mac,
                      lower_num_cap_mac,
                      upper_num_cap_mac,
                      lambda_fai_dur,
                      lower_fai_dur,
                      upper_fai_dur,
                      lower_fai_itv,
                      upper_fai_itv,
                      lambda_job_itv,
                      lower_job_itv,
                      upper_job_itv,
                      mu_lbd=0.01,
                      sigma_lbd=0.01,
                      lower_lbd=0.01,
                      upper_lbd=0.1,
                      ):
        # num of operation per job, ensure max num of operation < num_operation_max
        assert num_job * lower_num_opj <= num_operation_max, "invalid config!"
        num_operations = [lower_num_opj] * num_job
        num_ope_left = num_operation_max - lower_num_opj * num_job - 1
        for i in range(num_job):
            a = min(random.randint(0, upper_num_opj - lower_num_opj), num_ope_left)
            num_ope_left = num_ope_left - a
            num_operations[i] = num_operations[i] + a
        random.shuffle(num_operations)

        # split initial jobs and success jobs
        num_suc_job = int(num_job * dynamic_prop)
        num_init_job = num_job - num_suc_job

        init_job_list = []
        finish_time_min = 0
        for i in range(num_init_job):
            job = Job.generate_job(num_machine, num_operations[i], lower_pres_mu,
                                   upper_pres_mu, sigma_pres_time, lower_pres_time, upper_pres_time,
                                   mu_num_cap_mac, sigma_num_cap_mac, lower_num_cap_mac, upper_num_cap_mac)
            finish_time_min = max(finish_time_min, job.finish_time_min)
            init_job_list.append(job)

        suc_job_list = []
        occur_time = 0
        for i in range(num_suc_job):
            job = Job.generate_job(num_machine, num_operations[i], lower_pres_mu,
                                   upper_pres_mu, sigma_pres_time, lower_pres_time, upper_pres_time,
                                   mu_num_cap_mac, sigma_num_cap_mac, lower_num_cap_mac, upper_num_cap_mac)
            iterval = sample_truncated_exponential(lambda_job_itv, lower_job_itv, upper_job_itv)
            iterval = int(iterval.item())
            occur_time = occur_time + iterval
            suc_job_list.append((job, occur_time))

        broken_down_list = []
        for mac_id in range(num_machine):
            # sample lambda for each machine
            occur_time = 0
            lbd = sample_truncated_normal(mu_lbd, sigma_lbd, lower_lbd, upper_lbd).item()
            for _ in range(max_broken_down):
                iterval = int(sample_truncated_exponential(lbd, lower_fai_itv, upper_fai_itv).item())
                occur_time = occur_time + iterval
                fai_dur = int(sample_truncated_exponential(lambda_fai_dur, lower_fai_dur, upper_fai_dur).item())
                broken_down_list.append([mac_id, occur_time, fai_dur])
        broken_down_list = sorted(broken_down_list, key=lambda x: x[1])

        return Case(init_job_list, suc_job_list, broken_down_list, num_machine)


class CaseGenerator:
    def __init__(self,
                 num_machine,
                 num_job,
                 num_operation_max=200,
                 dynamic_prop=0.5,
                 max_break_down=10,
                 lower_num_opj=4,
                 upper_num_opj=7,
                 lower_pres_mu=10,
                 upper_pres_mu=25,
                 sigma_pres_time=1,
                 lower_pres_time=10,
                 upper_pres_time=25,
                 mu_num_cap_mac=1,
                 sigma_num_cap_mac=1,
                 lower_num_cap_mac=1,
                 upper_num_cap_mac=5,
                 mu_lbd=0.01,
                 sigma_lbd=0.005,
                 lower_lbd=0.001,
                 upper_lbd=0.1,
                 lambda_fai_dur=0.1,
                 lower_fai_dur=5,
                 upper_fai_dur=30,
                 lower_fai_itv=50,
                 upper_fai_itv=500,
                 lambda_job_itv=0.1,
                 lower_job_itv=3,
                 upper_job_itv=50):
        upper_num_cap_mac = min(num_machine, upper_num_cap_mac)
        assert num_job * lower_num_opj <= num_operation_max, "invalid config!"
        self.num_machine = num_machine
        self.num_operation_max = num_operation_max
        self.num_job = num_job
        self.dynamic_prop = dynamic_prop
        self.max_broken_down = max_break_down

        # num of operation per job
        self.lower_num_opj = lower_num_opj
        self.upper_num_opj = upper_num_opj

        # operation process time
        self.lower_pres_mu = lower_pres_mu,
        self.upper_pres_mu = upper_pres_mu,
        self.sigma_pres_time = sigma_pres_time
        self.lower_pres_time = lower_pres_time
        self.upper_pres_time = upper_pres_time

        # number of capable machine for each operation
        self.mu_num_cap_mac = mu_num_cap_mac
        self.sigma_num_cap_mac = sigma_num_cap_mac
        self.lower_num_cap_mac = lower_num_cap_mac
        self.upper_num_cap_mac = upper_num_cap_mac

        # failure duration
        self.lambda_fai_dur = lambda_fai_dur
        self.lower_fai_dur = lower_fai_dur
        self.upper_fai_dur = upper_fai_dur

        # failure interval
        self.mu_lbd = mu_lbd
        self.sigma_lbd = sigma_lbd
        self.lower_lbd = lower_lbd
        self.upper_lbd = upper_lbd
        self.lower_fai_itv = lower_fai_itv
        self.upper_fai_itv = upper_fai_itv

        # job arrive interval
        self.lambda_job_itv = lambda_job_itv
        self.lower_job_itv = lower_job_itv
        self.upper_job_itv = upper_job_itv

    def generate_cases(self, num_cases):
        cases_list = []
        for i in range(num_cases):
            case = Case.generate_case(num_machine=self.num_machine, num_operation_max=self.num_operation_max,
                                      num_job=self.num_job, dynamic_prop=self.dynamic_prop,
                                      max_broken_down=self.max_broken_down, lower_num_opj=self.lower_num_opj,
                                      upper_num_opj=self.upper_num_opj, lower_pres_mu=self.lower_pres_mu,
                                      upper_pres_mu=self.upper_pres_mu, sigma_pres_time=self.sigma_pres_time,
                                      lower_pres_time=self.lower_pres_time, upper_pres_time=self.upper_pres_time,
                                      mu_num_cap_mac=self.mu_num_cap_mac, sigma_num_cap_mac=self.sigma_num_cap_mac,
                                      lower_num_cap_mac=self.lower_num_cap_mac,
                                      upper_num_cap_mac=self.upper_num_cap_mac,
                                      lambda_fai_dur=self.lambda_fai_dur, lower_fai_dur=self.lower_fai_dur,
                                      upper_fai_dur=self.upper_fai_dur, lower_fai_itv=self.lower_fai_itv,
                                      upper_fai_itv=self.upper_fai_itv,
                                      lambda_job_itv=self.lambda_job_itv, lower_job_itv=self.lower_job_itv,
                                      upper_job_itv=self.upper_job_itv,
                                      mu_lbd=self.mu_lbd,
                                      sigma_lbd=self.sigma_lbd,
                                      lower_lbd=self.lower_lbd,
                                      upper_lbd=self.upper_lbd
                                      )
            cases_list.append(case)
        return cases_list

    @staticmethod
    def to_json(cases_list):
        return [case.to_json() for case in cases_list]

    @staticmethod
    def from_json(source):
        case_list = []
        for case in source:
            init_job_list = []
            for job in case["init_job_list"]:
                operation_list = []
                for op in job:
                    operation_list.append(Operation(op["available_machine"], op["operation_times"]))
                init_job_list.append(Job(operation_list))
            suc_job_list = []
            for job, time in case["suc_job_list"]:
                operation_list = []
                for op in job:
                    operation_list.append(Operation(op["available_machine"], op["operation_times"]))
                suc_job_list.append((Job(operation_list), time))
            case_list.append(Case(init_job_list, suc_job_list, case["broken_down_list"], case['num_machine']))
        return case_list


def gen_data(args):
    case_generator = CaseGenerator(args['num_machine'], args['num_job'], args['num_operation_max'],
                                   dynamic_prop=args['dynamic_prop'],
                                   sigma_pres_time=args['sigma_pres_time'],
                                   mu_lbd=args['mu_lbd'])
    cases = case_generator.generate_cases(args['num_cases'])
    cases = CaseGenerator.to_json(cases)
    json.dump(cases, open(f"./data/{args['data_name']}.json", "w"))
