require 'rest-client'
require 'nokogiri'
require 'fileutils'
require 'pathname'

DIRECTORY = "KNNOutputs"

def file_update(dir, fname)
  response = RestClient.post('http://cs156.caltech.edu/scoreboard', 
    file: File.new(fname),
    teamid: "cnjyqpyv",
    valset: 1 )

  doc = Nokogiri::HTML(response.body)
  str = doc.xpath("//h3")[1]
  m = / (\d)\.(\d*) /.match(str)

  padding = ""

  if m.nil?
    return
  end

  if m[1].to_i == 1
    padding += "1"
  end

  padding += m[2]
  padding += "0" * [5 - padding.length, 0].max

  new_name = File.basename(fname, File.extname(fname)) + "_#{padding}.txt"
  puts("Renaming #{fname} to #{new_name}")

  FileUtils.mv(fname, dir + "/" + new_name)
end

def dir_update(directory)
  Pathname.new(directory).children.each do |path|
    file_update(directory, path)
  end
end

dir_update(DIRECTORY)
