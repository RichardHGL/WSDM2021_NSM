import sys

filter_domain = ['music.release', 'authority.musicbrainz', '22-rdf-syntax-ns#type', 'book.isbn',
                 'common.licensed_object', 'tv.tv_series_episode', 'type.namespace', 'type.content',
                 'type.permission', 'type.object.key', 'type.object.permission', 'type.type.instance',
                 'topic_equivalent_webpage', 'dataworld.freeq']
filter_set = set(filter_domain)
input = "data/fb_en.txt"
output = "manual_fb_filter.txt"
f_in = open(input)
f_out = open(output, "w")
num_line = 0
num_reserve = 0
for line in f_in:
    splitline = line.strip().split("\t")
    num_line += 1
    if len(splitline) < 3:
        continue
    rel = splitline[1]
    flag = False
    for domain in filter_set:
        if domain in rel:
            flag = True
            break
    if flag:
        continue
    f_out.write(line)
    num_reserve += 1
    if num_line % 1000000 == 0:
        print(num_line, num_reserve)
f_in.close()
f_out.close()
print(num_line, num_reserve)