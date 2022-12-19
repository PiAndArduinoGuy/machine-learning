
def create_data_file(data_size, data_save_location):
    x_data = []
    y_data = []
    for x in range(data_size):
        x_data.append(x)
        y_data.append(2*x + 1)

    with open(data_save_location, "w") as data_file:
        content = ""
        for x_and_y_data_index in range(len(x_data)):
            content = content + str(x_data[x_and_y_data_index]) + "," + str(y_data[x_and_y_data_index]) + "\n"
        data_file.write(content)






if __name__ == '__main__':
    create_data_file(1000, "/home/quintin/Desktop/data.txt")


