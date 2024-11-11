import argparse
import json
import os
import queue
import sys
import threading
from PIL import Image as im
import numpy as np
from scipy.ndimage import gaussian_filter
import shutil
import time
from multiprocessing import Queue, Pool, Manager


def grayscale(image_array):
    red_channel = image_array[..., 0]
    green_channel = image_array[..., 1]
    blue_channel = image_array[..., 2]

    # Ponderisane vrednosti za RGB komponente
    grayscale_image = (red_channel * 0.299 + green_channel * 0.587 + blue_channel * 0.114)
    return grayscale_image.astype(np.uint8)


def gaussian_blur(image_array, sigma=1):

    # Primena Gaussian blur na R, G, B kanale
    red_channel = gaussian_filter(image_array[..., 0], sigma=sigma)
    green_channel = gaussian_filter(image_array[..., 1], sigma=sigma)
    blue_channel = gaussian_filter(image_array[..., 2], sigma=sigma)

    # Kombinovanje kanala nazad u jednu sliku
    blurred_image = np.zeros_like(image_array)
    blurred_image[..., 0] = red_channel
    blurred_image[..., 1] = green_channel
    blurred_image[..., 2] = blue_channel

    # Ukoliko nam postoji i alfa kanal,
    # potrebno je da i njega iskombinujemo kako bismo dobili RGBA kao što je bilo u originalnoj slici
    if image_array.shape[-1] == 4:
        alpha_channel = image_array[..., 3]
        blurred_image[..., 3] = alpha_channel

    blurred_image = np.clip(blurred_image, 0, 255)

    # Osigurajte da su vrednosti u validnom opsegu
    return blurred_image.astype(np.uint8)


def adjust_brightness(image_array, factor=1.0):
    mean_intensity = np.mean(image_array, axis=(0, 1), keepdims=True)  # Računanje srednje vrednosti piksela
    image_array = (image_array - mean_intensity) * factor + mean_intensity  # Skaliranje prema srednjoj vrednosti

    '''
    # Ručno implementirani clamp
    adjusted_image = np.where(image_array < 0, 0, image_array)  # Postavljanje vrednosti ispod 0 na 0
    adjusted_image = np.where(adjusted_image > 255, 255, adjusted_image)  # Postavljanje vrednosti iznad 255 na 255
    '''

    # Osiguravamo da vrednosti ostanu između 0 i 255
    adjusted_image = np.clip(image_array, 0, 255)

    return adjusted_image.astype(np.uint8)


def load_JSON_file(json_path, task_registry):
    if os.path.exists(json_path):
        with open(json_path) as f:
            params = json.load(f)
            image_id = params['image_id']
            filter_type = params['filter']
            output_path = params['output_path']
            task_params = params.get('params', {})

            with task_registry.lock:
                task_id = task_registry.counter.value
                task_registry.counter.value += 1

            task = Task(
                task_id = task_id,
                image_id = image_id,
                filter_type = filter_type,
                params = task_params,
                output_path = output_path,
                status = 'Waiting'
            )
        return task
    else:
        message_queue.put("JSON file path does not exist.")


def load_image(image_path):
    image = im.open(image_path)
    return np.array(image)


def save_image(image_array, output_path):
    image = im.fromarray(image_array)
    image.save(output_path)


class Image:
    def __init__(self,image_id, image_size, path, status, tasks , flag_delete, filters, processing_time, processed_size):
        self.image_id = image_id
        self.image_size = image_size
        self.path = path
        self.status = status
        self.tasks = tasks
        self.flag_delete = flag_delete
        self.filters = filters
        self.processing_time = processing_time
        self.processed_size = processed_size

    def add_task(self, task):
        self.tasks.append(task)


class Task:
    def __init__(self,task_id, image_id, filter_type, params, output_path, status):
        self.task_id = task_id
        self.image_id = image_id
        self.filter_type = filter_type
        self.params = params
        self.output_path = output_path
        self.status = status

    def __str__(self):
        return (f"Task ID: {self.task_id}, Image ID: {self.image_id}, Filter: {self.filter_type}")


class ImageRegistry:
    def __init__(self, manager):
        self.images = manager.list()
        self.counter = manager.Value('i', 0)
        self.lock = manager.Lock()


    def add_image(self, image):
        with self.lock:
            self.images.append(image)
            self.counter.value += 1


    def list_images(self):
        return self.images

    def describe_image(self, image_id):
        return self.images[image_id]


    def delete_image(self, image_id, task_registry):
        with self.lock:
            image_to_delete = None
            for image in self.images:
                if image.image_id == image_id:
                    image_to_delete = image
                    break

            if not image_to_delete:
                message_queue.put(f"Image with ID {image_id} not found!")
                return

            image_to_delete.flag_delete = True

            tasks_using_image = [task for task in task_registry.tasks.values()
                                 if task.image_id == image_id and task.status != 'Finished']

            if tasks_using_image:
                while any(task.status != 'Finished' for task in tasks_using_image):
                    with task_registry.condition:
                        task_registry.condition.wait(timeout=0.5)

            if os.path.exists(image_to_delete.path):
                try:
                    os.remove(image_to_delete.path)
                except Exception as e:
                    message_queue.put(f"Error deleting file {image_to_delete.path}: {e}")

            self.images = [img for img in self.images if img.image_id != image_id]




class TaskRegistry:
    def __init__(self, manager):
        self.tasks = manager.dict()
        self.counter = manager.Value('i', 0)
        self.lock = manager.Lock()
        self.condition = manager.Condition()

    def add_task(self, task):
        with self.lock:
            self.tasks[task.task_id] = task
            with self.condition:
                self.condition.notify_all()

    def get_task(self, task_id):
        return self.tasks.get(task_id)

    def update_task_status(self, task_id, status):
        if task_id in self.tasks:
            task = self.tasks[task_id]
            task.status = status
            self.tasks[task_id] = task



def process_image_task(image, task, image_registry, task_queue):
    try:
        image_array = load_image(image.path)
        start_time = time.time()

        if task.filter_type == 'grayscale':
            result_image = grayscale(image_array)
        elif task.filter_type == 'blur':
            result_image = gaussian_blur(image_array,**task.params)
        elif task.filter_type == 'brightness':
            result_image = adjust_brightness(image_array,**task.params)
        else:
            return None
    except Exception as e:
        message_queue.put(f"Error in process_image_task: {e}")

    save_image(result_image, task.output_path)
    processing_time = time.time() - start_time
    processing_size = os.path.getsize(task.output_path)

    new_image = Image(
        image_id=image_registry.counter.value,
        image_size= image.image_size,
        path=task.output_path,
        status='Processed',
        tasks=[task],
        flag_delete=False,
        filters=[task.filter_type],
        processing_time= processing_time,
        processed_size= processing_size,
    )

    if isinstance(image.tasks, list):
        for task in image.tasks:
            new_image.add_task(task)

    image_registry.add_image(new_image)

    return  task, task_queue


def task_completed(result):
    task, task_queue = result
    task_queue.put(task.task_id)


def copy_image(image_path):
    destination_folder = "images"
    image_name = os.path.basename(image_path)
    destination_path = os.path.join(destination_folder, image_name)


    if os.path.exists(destination_path):
        return destination_path

    shutil.copy(image_path, destination_path)
    return destination_path


def add_command(path, image_registry):
    if os.path.exists(path):
        destination_path = copy_image(path)
    else:
        message_queue.put(f"Path {path} does not exist!")
        return
    image = Image(
        image_id=image_registry.counter.value,
        image_size = os.path.getsize(path),
        path = destination_path,
        status = 'original',
        tasks = [],
        flag_delete = False,
        filters = [],
        processing_time = 0,
        processed_size = 0,
    )

    image_registry.add_image(image)


def process_command(json, task_registry,image_registry, task_queue, shutdown_event, pool):
    task = load_JSON_file(json, task_registry)
    task_registry.add_task(task)
    image = image_registry.images[int(task.image_id)]

    if image.flag_delete:
        message_queue.put("Image cant be used because it was marked for deletion")
    else:
        pool.apply_async(process_image_task, args=(image, task, image_registry,task_queue), callback=task_completed)


def list_command(image_registry, message_queue):
    images = image_registry.list_images()
    if images:
        for image in images:
            message_queue.put(f"Image ID: {image.image_id}, Path: {image.path}")
    else:
        message_queue.put("No images found!")


def describe_command(image_id, image_registry, message_queue):
    try:
        image = image_registry.describe_image(int(image_id))
        if image:
            tasks_str = ', '.join(str(task) for task in image.tasks)
            message_queue.put(
                f"Image ID: {image_id}, Status: {image.status}, "
                f"Tasks: [{tasks_str}], Filters: {image.filters}, "
                f"Size: {image.image_size}, Processed size: {image.processed_size}"
            )
        else:
            message_queue.put(f"No image found with ID {image_id}.")
    except IndexError:
        message_queue.put(f"Invalid image ID {image_id}.")



def delete_command(image_id,image_registry, message_queue, task_registry):
    image_registry.delete_image(int(image_id), task_registry)


def process_commands(input, image_registry,task_registry, message_queue, shutdown_event, task_queue, pool):
    string = input.split(' ')
    command = string[0]

    if command == 'add':
        if len(string) == 2:
            path = string[1]
            threading.Thread(target=add_command, args=(path, image_registry)).start()
        else:
            message_queue.put(f"Invalid command '{command}'!")


    elif command == 'process':
        if len(string) == 3:
            image_id = string[1]
            json = string[2]
            threading.Thread(target=process_command, args=(json, task_registry,image_registry, task_queue, shutdown_event, pool)).start()
        else:
            message_queue.put(f"Invalid command '{command}'!")


    elif command == 'list':
        threading.Thread(target=list_command, args=(image_registry, message_queue)).start()


    elif command == 'describe':
        if len(string) == 2:
            image_id = string[1]
            threading.Thread(target=describe_command, args=(image_id, image_registry, message_queue)).start()
        else:
            message_queue.put(f"Invalid command '{command}'!")


    elif command == 'delete':
        if len(string) == 2:
            image_id = string[1]
            threading.Thread(target=delete_command, args=(image_id,image_registry, message_queue, task_registry)).start()
        else:
            message_queue.put(f"Invalid command '{command}'!")


    elif command == 'exit':
        shutdown_event.set()

    else:
        message_queue.put("Wrong command!")


def message_printer(message_queue, shutdown_event):
    while not shutdown_event.is_set():
        while not message_queue.empty():
            message = message_queue.get()
            print(message)
        time.sleep(0.01)



def task_worker(task_registry,image_registry, task_queue, shutdown_event):
    while not shutdown_event.is_set():
        try:
            task_id = task_queue.get(timeout=1)

            with task_registry.lock:
                task = task_registry.get_task(task_id)

                if task is None:
                    message_queue.put(f"Task {task_id} not found!")
                    continue

                task.status = 'Finished'
                task_registry.update_task_status(task_id, 'Finished')

            with task_registry.condition:
                task_registry.condition.notify_all()

        except queue.Empty:
            continue
        except EOFError:
            break




if __name__ == '__main__':
    with Manager() as manager:
        image_registry = ImageRegistry(manager)
        task_registry = TaskRegistry(manager)
        message_queue = manager.Queue()
        shutdown_event = threading.Event()
        task_queue = manager.Queue()
        pool = Pool(processes=4)

        task_thread = threading.Thread(target=task_worker, args=(task_registry,image_registry, task_queue, shutdown_event))
        message_thread = threading.Thread(target=message_printer, args=(message_queue, shutdown_event))
        task_thread.start()
        message_thread.start()

        while not shutdown_event.is_set():
            try:
                command = input("> ").strip()
                process_commands(command, image_registry,task_registry, message_queue, shutdown_event, task_queue, pool)
            except EOFError:
                break

        task_thread.join()
        message_thread.join()
        pool.close()
        pool.join()


