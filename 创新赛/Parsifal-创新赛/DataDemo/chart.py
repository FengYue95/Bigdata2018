#!/usr/bin/env python
# coding=utf-8
import tornado.web
import os
import sys
import json
import socket
import tornado.ioloop
import tornado.options
import tornado.httpserver
from tornado.options import define, options


# 对应index地址的方法
class IndexHandler(tornado.web.RequestHandler):
    def get(self):
        self.render("index.html")


class LifeHandler(tornado.web.RequestHandler):
    def get(self):
        self.render("life.html")

class SocialHandler(tornado.web.RequestHandler):
    def get(self):
        self.render("social.html")

class downLoadHandler(tornado.web.RequestHandler):
    def get(self):
        self.set_header('Content-Type', 'application/octet-stream')
        self.set_header('Content-Disposition', 'attachment; filename=Parsifal.pptx')
        # 读取的模式需要根据实际情况进行修改
        with open("Parsifal.pptx", 'rb') as f:
            while True:
                data = f.read()
                if not data:
                    break
                self.write(data)
        self.finish()


# 设置地址映射
url = [
    (r'/index', IndexHandler),
    (r'/life', LifeHandler),
    (r'/social', SocialHandler),
    (r'/download', downLoadHandler)

]
# 设置路径
settings = dict(
    template_path=os.path.join(os.path.dirname(__file__), "templates"),
    static_path=os.path.join(os.path.dirname(__file__), "statics")
)
# 配置application
application = tornado.web.Application(
    handlers=url,
    **settings
)

define("port", default=8999, help="run on the given port", type=int)


def main():
    tornado.options.parse_command_line()
    http_server = tornado.httpserver.HTTPServer(application)
    http_server.listen(options.port)
    tornado.ioloop.IOLoop.instance().start()


if __name__ == "__main__":
    main()
