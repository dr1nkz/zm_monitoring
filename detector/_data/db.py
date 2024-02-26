import mysql.connector
import argparse
# Acceptable formats include hostname
# ZM_DB_HOST=localhost
# ZoneMinder database user
# ZM_DB_USER=zmuser
# ZoneMinder database password
# ZM_DB_PASS=zmpass
# ZoneMinder database name
# ZM_DB_NAME=zm


def parseurl(temp):
    if '://' in temp[-1]:
        temp = temp[-1]
    else:
        temp = temp[0] + '://' + temp[1] + ':' + temp[2] + temp[3]
    return temp


def parse_monitor_parameters(monitor_id):
    mydb = mysql.connector.connect(
        host='localhost',
        user='zmuser',
        password='zmpass',
        database='zm'
    )

    # ------Address------
    mycursor = mydb.cursor()

    mycursor.execute(
        f"SELECT Protocol, Host, Port, Path FROM Monitors WHERE ID={monitor_id}")

    address = mycursor.fetchall()[0]
    address = parseurl(address)
    print(address, '\n')

    # ------Zones------
    mycursor.execute(f"SELECT Coords FROM Zones WHERE MonitorId={monitor_id}")

    zones = mycursor.fetchall()
    print(zones)

    return address, zones


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('monitor_id')
    args = parser.parse_args()
    monitor_id = args.monitor_id
    parse_monitor_parameters(monitor_id)
