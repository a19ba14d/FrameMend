"""Run a fresh watermark repair using the existing command-line stages."""

import argparse
from contextlib import contextmanager
from errno import EACCES, EAGAIN, EDEADLK
from hashlib import file_digest
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from queue import Empty, Queue
from threading import Thread
from time import monotonic, perf_counter, sleep
from wm_runtime import WorkflowLimit, bounded_timeout, check_budget


ENTRY = Path(__file__).with_name('watermark.py')
PATCH_COMMANDS = {
    'local-patch', 'url-reveal', 'animation-template', 'animation-patch',
    'wipe-template', 'wipe-patch', 'caption-inpaint', 'corner-trim', 'pose-patch', 'region-patch',
}


def read_object(path):
    data = json.loads(path.read_text(encoding='utf-8-sig'))
    if not isinstance(data, dict):
        raise ValueError(f'Expected a JSON object in {path}.')
    return data


def stage_args(value, paths):
    if not isinstance(value, list) or any(not isinstance(arg, str) for arg in value):
        raise ValueError('Options must be a JSON list of individual arguments.')
    return [arg.format_map(paths) for arg in value]


def has_option(args, options):
    return any(arg.split('=', 1)[0] in options for arg in args)


def inside(directory, value):
    path = (directory/value).resolve()
    if path==directory or not path.is_relative_to(directory):
        raise ValueError(f'Output must be inside the new job directory: {value}')
    return path


def build_plan(source, directory, recipe_path, cpu_decode):
    recipe = read_object(recipe_path)
    profile_path = (recipe_path.parent/recipe['profile']).resolve(strict=True)
    profile = read_object(profile_path)
    paths = dict(recipe=str(recipe_path.parent), work=str(directory))
    common = ['--profile', str(profile_path), '--workdir', str(directory)]
    if cpu_decode:
        common.append('--cpu-decode')
    shared_render = stage_args(recipe.get('shared_render', []), paths)
    final_render = stage_args(recipe.get('final_render', []), paths)
    reference_render = stage_args(recipe.get('reference_render', ['--edges', 'spatial', '--cq', '16']), paths)
    reserved = {'--profile', '--workdir', '--frames', '--animation-patch', '--caption-patch', '--pairs'}
    if has_option(shared_render+final_render+reference_render, reserved):
        raise ValueError('Render options must describe full-length rendering. Declare patches as separate steps.')

    plan = []
    outputs = set()

    def add(name, command, args, files):
        if outputs.intersection(files):
            raise ValueError(f'A planned output has already been assigned in {name}.')
        outputs.update(files)
        plan.append(dict(name=name, args=[command, *map(str, args)], outputs=[str(file) for file in files]))

    add('scan', 'scan', [source, *common], [directory/'track.npz'])
    if profile.get('translucent') or profile.get('stationary'):
        options = []
        if recipe.get('calibration_pairs'):
            options = ['--pairs', (recipe_path.parent/recipe['calibration_pairs']).resolve(strict=True)]
        add('calibrate', 'calibrate', [source, *common, *options], [directory/'matte.npz'])
    prepared = [directory/'corners'/'manifest.json'] if profile.get('corners') else []
    if has_option(shared_render, {'--opening'}):
        prepared.append(directory/'opening.npz')
    add('prepare', 'prepare', [source, *common, *shared_render], prepared)

    steps = recipe.get('patches', [])
    if not isinstance(steps, list) or any(not isinstance(step, dict) for step in steps):
        raise ValueError('patches must be a list of step objects.')
    reference = directory/'reference.mp4'
    if any(step.get('reference') for step in steps):
        add('build reference', 'render', [source, reference, *common, *shared_render, *reference_render], [reference])
    compositing = []
    for step in steps:
        command = step['command']
        if command not in PATCH_COMMANDS:
            raise ValueError(f'Unsupported patch command {command!r}.')
        if type(step.get('reference', False)) is not bool:
            raise ValueError('reference must be a JSON boolean.')
        config_path = (recipe_path.parent/step['config']).resolve(strict=True)
        attachment = step.get('attach')
        if attachment not in (None, 'animation', 'caption'):
            raise ValueError('attach must be animation or caption.')
        selection = step.get('jobs')
        if selection is None:
            jobs = [None]
        else:
            available = read_object(config_path).get('jobs', {})
            if not isinstance(available, dict):
                raise ValueError(f'jobs in {config_path.name} must be an object.')
            jobs = list(available) if selection=='all' else selection
            if (not isinstance(jobs, list) or not jobs or any(not isinstance(job, str) or job not in available
                                                            for job in jobs) or len(set(jobs))!=len(jobs)):
                raise ValueError(f'Choose distinct jobs from {config_path.name}.')
        for job in jobs:
            values = dict(paths, job=job or '')
            output = inside(directory, step['output'].format_map(values))
            if output.suffix.lower()!='.npz':
                raise ValueError('Patch stages must output .npz files.')
            base = [reference] if step.get('reference') else []
            selected = ['--job', job] if job else []
            options = stage_args(step.get('args', []), values)
            if has_option(options, {'--profile', '--workdir', '--config', '--job'}):
                raise ValueError('Patch options must use the workflow profile, job directory and declared config.')
            add(step.get('name', output.stem).format_map(values), command,
                [source, *base, output, *common, '--config', config_path, *selected, *options], [output])
            if attachment:
                compositing.extend([f'--{attachment}-patch', output])

    final = directory/'result.mp4'
    verification = directory/'verification.json'
    audit = directory/'watermark-audit.json'
    rendered = [final]
    if has_option(shared_render+final_render, {'--recover-corner-overlap'}):
        rendered.append(directory/'track-recovered.npz')
    add('render', 'render', [source, final, *common, *shared_render, *final_render, *compositing], rendered)
    add('verify', 'verify', [source, final, '--report', verification], [verification])
    add('audit', 'audit', [source, final, *common, '--report', audit], [audit])
    for check in recipe.get('animation_checks', []):
        name = check['name']
        report = inside(directory, f'{name}-audit.json')
        options = ['--cpu-decode'] if cpu_decode else []
        add(f'audit {name}', 'animation-audit',
            [source, final, '--starts', *check['starts'], '--ages', *check['ages'], '--region', *check['region'],
             *options, '--report', report], [report])
        plan[-1]['occurrences'] = check['occurrences']
    return plan


def write_object(path, data):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    for attempt in range(8):
        try:
            tmp.replace(path)
            return
        except PermissionError:
            if attempt==7:
                raise
            # A reader or Windows indexer may hold the report briefly.
            sleep(.05*(attempt+1))


def write_state(directory, state):
    write_object(directory/'run.json', state)


def fingerprints(files):
    hashes = {}
    for path in sorted(set(map(Path, files))):
        with path.open('rb') as stream:
            hashes[str(path)] = file_digest(stream, 'sha256').hexdigest()
    return hashes


def argument_path(arg):
    return Path(arg.split('=', 1)[-1] if arg.startswith('--') else arg)


def input_files(source, recipe, directory, plan):
    inputs = {source, recipe}
    for stage in plan:
        for arg in stage['args']:
            path = argument_path(arg)
            if path.is_absolute() and not path.is_relative_to(directory) and path.is_file():
                inputs.add(path)
    return inputs


def lock_job(directory):
    stream = (directory/'.workflow.lock').open('a+b')
    try:
        if stream.tell()==0:
            stream.write(b'\0')
            stream.flush()
        stream.seek(0)
        if os.name=='nt':
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as error:
        stream.close()
        raise ValueError('Another workflow is using this job directory.') from error
    return stream


class CapacityTimeout(TimeoutError):
    """All slots remained occupied until the admission deadline."""


@contextmanager
def capacity_slot(root, count, wait_seconds, *, waiting=None):
    directories = [root/str(i) for i in range(count)]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
    started = monotonic()
    heartbeat = started+15
    while True:
        check_budget()
        for index, directory in enumerate(directories, 1):
            try:
                lock = lock_job(directory)
            except ValueError as error:
                if not isinstance(error.__cause__, OSError) or error.__cause__.errno not in (EACCES, EAGAIN, EDEADLK):
                    raise
                continue
            with lock:
                yield index, monotonic()-started
            return
        now = monotonic()
        if now-started >= wait_seconds:
            raise CapacityTimeout('No shared processing slot became available.')
        if waiting is not None and now >= heartbeat:
            waiting(now-started)
            heartbeat = now+15
        sleep(.2)


def resume_state(directory, plan, inputs):
    state = read_object(directory/'run.json')
    stages = state['stages']
    fields = ('name', 'args', 'outputs', 'occurrences')
    signature = lambda items: [[stage.get(key) for key in fields] for stage in items]
    if signature(stages)!=signature(plan) or state.get('inputs')!=inputs:
        raise ValueError('The source, configuration or plan has changed. Choose a new job directory.')
    first = next((i for i, stage in enumerate(stages) if stage.get('status')!='complete'), len(stages))
    if any(stage.get('status')=='complete' for stage in stages[first:]):
        raise ValueError('The run record must contain a consecutive prefix of completed stages.')
    for stage in stages[:first]:
        artifacts = stage.get('artifacts', {})
        if not set(stage['outputs']).issubset(artifacts) or fingerprints(artifacts)!=artifacts:
            raise ValueError(f'A completed artifact changed in {stage["name"]}.')
    return state, first


def stage_files(directory, stage):
    files = list(map(Path, stage['outputs']))
    if directory/'corners'/'manifest.json' in files:
        files.extend(sorted((directory/'corners').glob('*.npy')))
    return files


def reuse_prepared(directory, state, cache, trace):
    previous = read_object(cache/'run.json')

    def canonical(value, relocate=False):
        path = argument_path(value)
        if not path.is_absolute():
            return value
        if relocate and path.is_relative_to(cache):
            path = directory/path.relative_to(cache)
        prefix = value.split('=', 1)[0]+'=' if value.startswith('--') else ''
        return prefix+str(path)

    first = 0
    for stage, old in zip(state['stages'], previous['stages']):
        if stage['args'][0] == 'render' or old.get('status') != 'complete':
            break
        if ([canonical(arg, relocate=True) for arg in old['args']] != [canonical(arg) for arg in stage['args']]
                or [canonical(path, relocate=True) for path in old['outputs']] != stage['outputs']):
            break
        inputs = set(map(str, map(argument_path, stage['args']))) & state['inputs'].keys()
        if any(state['inputs'][arg] != previous.get('inputs', {}).get(arg) for arg in inputs):
            break
        files = stage_files(cache, old)
        if any(not path.resolve().is_relative_to(cache) for path in files):
            raise ValueError('A cached file is outside its original job directory.')
        actual = fingerprints(files)
        recorded = old.get('artifacts', {})
        if not set(old['outputs']).issubset(recorded) or any(actual.get(path) != sha for path, sha in recorded.items()):
            raise ValueError(f'A cached artifact changed in {old["name"]}.')
        for path in files:
            target = directory/path.relative_to(cache)
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(path, target)
            except OSError:
                if target.exists():
                    raise
                shutil.copyfile(path, target)
        artifacts = {canonical(path, relocate=True): sha for path, sha in actual.items()}
        stage.update(status='complete', seconds=0, artifacts=artifacts, reused_from=str(cache))
        trace('stage.reused', stage=stage['name'], from_directory=str(cache), artifacts=artifacts)
        first += 1
    trace('cache.checked', from_directory=str(cache), reused_stages=first)
    return first


class StageFailed(subprocess.CalledProcessError):
    def __init__(self, returncode, command, stage, last_line):
        super().__init__(returncode, command, output=last_line)
        self.stage = stage

    def __str__(self):
        detail = f' {self.output}' if self.output else ''
        return f'{self.stage} failed with exit code {self.returncode}.{detail}'


def worker_lines(worker):
    messages = Queue()

    def read():
        try:
            for line in worker.stdout:
                messages.put(line)
        finally:
            messages.put(None)

    Thread(target=read, daemon=True).start()
    while True:
        check_budget()
        try:
            line = messages.get(timeout=.25)
        except Empty:
            continue
        if line is None:
            return
        yield line


def stop_worker(worker):
    if worker.poll() is not None:
        return
    if os.name == 'nt':
        subprocess.run(['taskkill', '/PID', str(worker.pid), '/T', '/F'],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10,
                       creationflags=subprocess.CREATE_NO_WINDOW)
    else:
        import signal
        os.killpg(worker.pid, signal.SIGKILL)
    if worker.poll() is None:
        worker.kill()


def execute_plan(directory, state, first, trace=None, *, render_slots=1):
    plan = state['stages']
    state['status'] = 'running'
    for number, stage in enumerate(plan[first:], first+1):
        check_budget()
        stage.pop('error', None)
        stage['status'] = 'running'
        write_state(directory, state)
        started = perf_counter()
        print(f'[{number}/{len(plan)}] {stage["name"]}', flush=True)
        name = stage['name']
        command = [sys.executable, '-X', 'utf8', '-u', str(ENTRY), *stage['args']]
        if trace:
            trace('stage.started', stage=name, number=number, total=len(plan), command=command,
                  outputs=stage['outputs'])
        try:
            for file in map(Path, stage['outputs']):
                file.parent.mkdir(parents=True, exist_ok=True)
            environment = ({**os.environ, 'WATERMARK_RENDER_SLOTS': str(render_slots)}
                           if name == 'render' else None)
            if trace:
                with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                      text=True, encoding='utf-8', errors='replace', env=environment,
                                      creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0,
                                      start_new_session=os.name != 'nt') as worker:
                    try:
                        last_line = None
                        for line in worker_lines(worker):
                            if line := line.rstrip():
                                last_line = line
                                print(line, flush=True)
                                trace('stage.output', stage=name, line=line)
                        if worker.wait(timeout=bounded_timeout(30)):
                            raise StageFailed(worker.returncode, command, name, last_line)
                    finally:
                        if worker.poll() is None:
                            stop_worker(worker)
            else:
                subprocess.run(command, check=True, env=environment, timeout=bounded_timeout(86400))
            for file in map(Path, stage['outputs']):
                if not file.is_file():
                    raise RuntimeError(f'Expected output missing: {file}')
            if 'occurrences' in stage:
                result = read_object(Path(stage['outputs'][0]))
                if len(result['source_matches'])!=stage['occurrences'] or result['output_matches']:
                    raise RuntimeError(f'Animation audit failed. See {stage["outputs"][0]}.')
            stage['artifacts'] = fingerprints(stage_files(directory, stage))
            stage['status'] = 'complete'
        except (OSError, RuntimeError, subprocess.CalledProcessError, subprocess.TimeoutExpired, KeyboardInterrupt) as error:
            stage['status'] = 'failed'
            stage['error'] = str(error)
            state['status'] = 'failed'
            if trace:
                trace('stage.error', stage=name, error_type=type(error).__name__, error=str(error))
            if isinstance(error, subprocess.TimeoutExpired):
                raise WorkflowLimit(f'The {name} stage exceeded its available processing time.') from error
            raise
        finally:
            stage['seconds'] = round(perf_counter()-started, 2)
            write_state(directory, state)
            if trace:
                trace('stage.finished', stage=name, status=stage['status'], seconds=stage['seconds'],
                      artifacts=stage.get('artifacts', {}))
    state['status'] = 'complete'
    write_state(directory, state)
    print(f'Workflow complete. Results in {directory}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path, help='Original video.')
    parser.add_argument('job', type=Path, help='Directory for the full run.')
    parser.add_argument('--recipe', type=Path, required=True)
    parser.add_argument('--cpu-decode', action='store_true')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--plan', action='store_true', help='Print the stages without running them.')
    mode.add_argument('--resume', action='store_true', help='Continue a recorded job with the same inputs and plan.')
    args = parser.parse_args()
    source = args.source.resolve(strict=True)
    recipe = args.recipe.resolve(strict=True)
    directory = args.job.resolve()
    if not source.is_file() or (args.resume and not directory.is_dir()) or (not args.resume and directory.exists()):
        raise ValueError('Supply an original video and a new job directory, or use --resume for an existing job.')
    plan = build_plan(source, directory, recipe, args.cpu_decode)
    if args.plan:
        print(json.dumps(plan, indent=2))
        return
    inputs = fingerprints(input_files(source, recipe, directory, plan))
    if not args.resume:
        directory.mkdir(parents=True)
    with lock_job(directory):
        if args.resume:
            state, first = resume_state(directory, plan, inputs)
            print(f'Validated {first} completed stages before resuming.', flush=True)
        else:
            first = 0
            state = dict(source=str(source), recipe=str(recipe), inputs=inputs, status='running', stages=plan)
        execute_plan(directory, state, first)


if __name__=='__main__':
    try:
        main()
    except (KeyError, TypeError, ValueError, RuntimeError, OSError, subprocess.CalledProcessError) as error:
        print(f'Error: {error}', file=sys.stderr)
        sys.exit(1)
