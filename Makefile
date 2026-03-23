loc:
	find src tests mm-demo/src -name '*.rs' | xargs wc -l

gitaddall:
	git add src tests 
